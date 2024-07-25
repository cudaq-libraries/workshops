# pip install openfermionpyscf
# python H2-full-space.py

import cudaq 

hydrogen_count = 2
bond_distance = 0.7474
geometry = [('H', (0, 0, i * bond_distance)) for i in range(hydrogen_count)]

molecule, data = cudaq.chemistry.create_molecular_hamiltonian(geometry, 'sto-3g', 1, 0)
electron_count = data.n_electrons
qubit_count = 2 * data.n_orbitals

@cudaq.kernel
def kernel(thetas: list[float]):
    qubits = cudaq.qvector(qubit_count)
    
    # Prepare the Hartree Fock State.
    for i in range(electron_count):
        x(qubits[i])

    # UCCSD ansatz
    cudaq.kernels.uccsd(qubits, thetas, electron_count, qubit_count)

parameter_count = cudaq.kernels.uccsd_num_parameters(electron_count, qubit_count)
optimizer = cudaq.optimizers.COBYLA()
energy, parameters = cudaq.vqe(kernel, molecule, optimizer, parameter_count=parameter_count)
print(energy)
