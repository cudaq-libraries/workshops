import cudaq
import numpy as np
import scipy

# Single-node, single gpu
cudaq.set_target("nvidia")
multi_gpu = False

# Single-node, multi-GPU
#cudaq.set_target("nvidia", option='mqpu,fp64')
#multi_gpu = True

# Define H2 molecule
geometry = [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7474))]

hamiltonian, data = cudaq.chemistry.create_molecular_hamiltonian(
    geometry, 'sto-3g', 1, 0)

electron_count = data.n_electrons
qubits_num = 2 * data.n_orbitals

spin_ham_matrix = hamiltonian.to_matrix()
e, c = np.linalg.eig(spin_ham_matrix)

# Find the ground state energy and the corresponding eigenvector
print('Ground state energy (classical simulation)= ', np.min(e), ', index= ',
      np.argmin(e))
min_indices = np.argmin(e)

# Eigen vector can be used to initialize the qubits
vec = c[:, min_indices]

# Collect coefficients from a spin operator so we can pass them to a kernel
def termCoefficients(ham: cudaq.SpinOperator) -> list[complex]:
    result = []
    ham.for_each_term(lambda term: result.append(term.get_coefficient()))
    return result


# Collect Pauli words from a spin operator so we can pass them to a kernel
def termWords(ham: cudaq.SpinOperator) -> list[str]:
    result = []
    ham.for_each_term(lambda term: result.append(term.to_string(False)))
    return result


coefficient = termCoefficients(hamiltonian)
pauli_string = termWords(hamiltonian)

@cudaq.kernel
def U_psi(qubits: cudaq.qview, dt: float, coefficients: list[complex],
          words: list[cudaq.pauli_word]):
    # Compute U_m = exp(-i m dt H)
    for i in range(len(coefficients)):
        exp_pauli(dt * coefficients[i].real, qubits, words[i])


@cudaq.kernel
def U_phi(qubits: cudaq.qview, dt: float, coefficients: list[complex],
          words: list[cudaq.pauli_word]):
    # Compute U_n = exp(-i n dt H)
    for i in range(len(coefficients)):
        exp_pauli(dt * coefficients[i].real, qubits, words[i])


@cudaq.kernel
def apply_pauli(qubits: cudaq.qview, word: list[int]):

    # Add H (Hamiltonian operator)
    for i in range(len(word)):
        if word[i] == 1:
            x(qubits[i])
        if word[i] == 2:
            y(qubits[i])
        if word[i] == 3:
            z(qubits[i])


@cudaq.kernel
def qfd_kernel(dt_alpha: float, dt_beta: float, coefficients: list[complex],
               words: list[cudaq.pauli_word], word_list: list[int],
               vec: list[complex]):

    ancilla = cudaq.qubit()
    qreg = cudaq.qvector(vec)

    h(ancilla)

    x(ancilla)
    cudaq.control(U_psi, ancilla, qreg, dt_alpha, coefficients, words)
    x(ancilla)

    cudaq.control(apply_pauli, ancilla, qreg, word_list)
    cudaq.control(U_phi, ancilla, qreg, dt_beta, coefficients, words)
    
def pauli_str(pauli_word, qubits_num):

    my_list = []
    for i in range(qubits_num):
        if str(pauli_word[i]) == 'I':
            my_list.append(0)
        if str(pauli_word[i]) == 'X':
            my_list.append(1)
        if str(pauli_word[i]) == 'Y':
            my_list.append(2)
        if str(pauli_word[i]) == 'Z':
            my_list.append(3)
    return my_list

# Define the spin-op x for real component and y for the imaginary component.

x_0 = cudaq.spin.x(0)
y_0 = cudaq.spin.y(0)

#Define parameters for the quantum Krylov space

dt = 0.5

# Dimension of the Krylov space
m_qfd = 4

# Compute the basis overlap matrix

## Single GPU:
if not multi_gpu:

    # Add identity operator to compute overlap of basis

    observe_op = 1.0
    for m in range(qubits_num):
        observe_op *= cudaq.spin.i(m)

    identity_word = observe_op.to_string(False)

    pauli_list = pauli_str(identity_word, qubits_num)
    #print(pauli_list)

    wf_overlap = np.zeros((m_qfd, m_qfd), dtype=complex)

    for m in range(m_qfd):
        dt_m = dt * m
        for n in range(m, m_qfd):
            dt_n = dt * n
            results = cudaq.observe(qfd_kernel, [x_0, y_0], dt_m, dt_n,
                                    coefficient, pauli_string, pauli_list, vec)
            temp = [result.expectation() for result in results]
            wf_overlap[m, n] = temp[0] + temp[1] * 1j
            if n != m:
                wf_overlap[n, m] = np.conj(wf_overlap[m, n])


else:

    ## Multi-GPU

    # Compute the basis overlap matrix

    # Add identity operator to compute overlap of basis

    observe_op = 1.0
    for m in range(qubits_num):
        observe_op *= cudaq.spin.i(m)

    identity_word = observe_op.to_string(False)

    pauli_list = pauli_str(identity_word, qubits_num)
    #print(pauli_list)

    wf_overlap = np.zeros((m_qfd, m_qfd), dtype=complex)

    collect_overlap_real = []
    collect_overlap_img = []

    count=0
    for m in range(m_qfd):
        dt_m = dt * m
        for n in range(m, m_qfd):
            dt_n = dt * n

            count_id=count%2
            #print(count_id)
            collect_overlap_real.append(cudaq.observe_async(qfd_kernel, x_0, dt_m, dt_n,
                                    coefficient, pauli_string, pauli_list, vec, qpu_id=count_id))

            collect_overlap_img.append(cudaq.observe_async(qfd_kernel, y_0, dt_m, dt_n,
                                    coefficient, pauli_string, pauli_list, vec, qpu_id=count_id+2))
            count += 1

    tot_dim = 0

    for n in range(m_qfd):
        for m in range(n,m_qfd):
            observe_result = collect_overlap_real[tot_dim].get()
            real_val = observe_result.expectation()

            observe_result=collect_overlap_img[tot_dim].get()
            img_val=observe_result.expectation()

            wf_overlap[m, n] = real_val + img_val * 1j
            if n != m:
                wf_overlap[n, m] = np.conj(wf_overlap[m, n])

            tot_dim += 1

         
# Compute the matrix Hamiltonian

## Single GPU:
if not multi_gpu:
    ham_matrx = np.zeros((m_qfd, m_qfd), dtype=complex)

    for m in range(m_qfd):
        dt_m = dt * m
        for n in range(m, m_qfd):
            dt_n = dt * n

            tot_e = np.zeros(2)
            for coef, word in zip(coefficient, pauli_string):
                #print(coef,word)

                pauli_list = pauli_str(word, qubits_num)
                #print(pauli_list)

                results = cudaq.observe(qfd_kernel, [x_0, y_0], dt_m, dt_n,
                                        coefficient, pauli_string, pauli_list, vec)

                temp = [result.expectation() for result in results]
                #print(temp)
                temp[0] = coef.real * temp[0]
                temp[1] = coef.imag * temp[1]

                tot_e[0] += temp[0]
                tot_e[1] += temp[1]

            ham_matrx[m, n] = tot_e[0] + tot_e[1] * 1j
            if n != m:
                ham_matrx[n, m] = np.conj(ham_matrx[m, n])

else:
    ## Multi-GPU

    ham_matrx = np.zeros((m_qfd, m_qfd), dtype=complex)


    for m in range(m_qfd):
        dt_m = dt * m
        for n in range(m, m_qfd):
            dt_n = dt * n

            ham_matrix_real = []
            ham_matrix_imag = []

            count=0
            tot_e = np.zeros(2)
            for coef, word in zip(coefficient, pauli_string):
                count_id=count%2
                #print(coef,word)

                pauli_list = pauli_str(word, qubits_num)
                #print(pauli_list)

                ham_matrix_real.append(cudaq.observe_async(qfd_kernel, x_0, dt_m, dt_n,
                                        coefficient, pauli_string, pauli_list, vec, qpu_id=count_id))
                ham_matrix_imag.append(cudaq.observe_async(qfd_kernel, y_0, dt_m, dt_n,
                                        coefficient, pauli_string, pauli_list, vec, qpu_id=count_id+2))

                count += 1

            i = 0
            for coef in coefficient:

                observe_result = ham_matrix_real[i].get()
                real_val = observe_result.expectation()

                observe_result=ham_matrix_imag[i].get()
                img_val=observe_result.expectation()

                tot_e[0] += real_val * coef.real
                tot_e[1] += img_val * coef.imag

                i+=1

            ham_matrx[m, n] = tot_e[0] + tot_e[1] * 1j
            if n != m:
                ham_matrx[n, m] = np.conj(ham_matrx[m, n])

# Diagonalize the matrix


def eig(H, s):
    #Solver for generalized eigenvalue problem

    # HC = SCE

    THRESHOLD = 1e-20
    s_diag, u = scipy.linalg.eig(s)
    s_prime = []
    for sii in s_diag:
        if np.imag(sii) > 1e-7:
            raise ValueError(
                "S may not be hermitian, large imag. eval component.")
        if np.real(sii) > THRESHOLD:
            s_prime.append(np.real(sii))

    X_prime = np.zeros((len(s_diag), len(s_prime)), dtype=complex)

    for i in range(len(s_diag)):
        for j in range(len(s_prime)):
            X_prime[i][j] = u[i][j] / np.sqrt(s_prime[j])

    H_prime = (((X_prime.conjugate()).transpose()).dot(H)).dot(X_prime)
    e_prime, C_prime = scipy.linalg.eig(H_prime)
    C = X_prime.dot(C_prime)

    return e_prime, C

eigen_value, eigen_vect = eig(ham_matrx[0:m_qfd, 0:m_qfd], wf_overlap[0:m_qfd,
                                                                      0:m_qfd])
print('Energy from QFD:')
print(np.min(eigen_value))