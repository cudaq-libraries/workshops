# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Hadamard Test
#
# Consider the observable $O$ and two generic quantum states $| \psi\rangle$ and $| \phi\rangle$. We want to calculate the quantity
# $$
# \langle \psi | O | \psi\rangle.
# $$
# where $O$ is a Pauli operator.
#
# First of all we shall prepare the states $|\psi\rangle$ and $|\phi\rangle$ using a quantum circuit for each of them. So we  have
# $$
# | \psi\rangle = U_{\psi}|0\rangle \qquad |\phi\rangle = U_{\phi}|0\rangle
# $$
#
# Let's define an observable we want to use:
# $$
# O = X_1X_2
# $$
#
# Now we can evaluate the matrix element using the following fact:
# $$
# \langle \psi|O|\phi\rangle = \langle 0 |U_\psi^\dagger O U_\phi |0\rangle
# $$
# This is just an expectation value which can be solved with a simple Hadamard test. The probability to measure $0$ or $1$ in the ancilla qubit is
#
# $$
# P(0) = \frac{1}{2} \left[ I + \operatorname{Re} \langle \psi| O |\phi\rangle \right]
# $$
#
# $$
# P(1) = \frac{1}{2} \left[ I - \operatorname{Re} \langle \psi| O | \phi\rangle \right]
# $$
#
# The difference between the probability of $0$ and $1$ gives
#
# $$
# \langle X\rangle = P(0)-P(1) = \operatorname{Re} \langle \psi | O |\phi\rangle.
# $$
#
# Similarly, the imaginary part can be obtained from Y measurement
# $$
# \langle Y\rangle = \operatorname{Im} \langle \psi | O | \phi\rangle.
# $$
#
# Combining these results, the quantity $\langle \psi | O | \psi\rangle$ is obtained.

# ### Numerical result as a reference:

import numpy as np

import cudaq


num_qubits = 2


@cudaq.kernel
def psi(num_qubits: int):
    q = cudaq.qvector(num_qubits)
    h(q[1])


@cudaq.kernel
def phi(num_qubits: int):
    q = cudaq.qvector(num_qubits)
    x(q[0])


psi_state = cudaq.get_state(psi, num_qubits)
print("Psi state: ", np.array(psi_state))

phi_state = cudaq.get_state(phi, num_qubits)
print("Phi state: ", np.array(phi_state))

ham = cudaq.spin.x(0) * cudaq.spin.x(1)
ham_matrix = ham.to_matrix()
print("hamiltonian: ", np.array(ham_matrix), "\n")

num_ev = np.array(psi_state).conj() @ ham_matrix @ np.array(phi_state).T

print("Numerical expectation value: ", num_ev)

# ### Using ``observe`` algorithmic primitive to compute the expectation value for ancilla qubits.

import cudaq


@cudaq.kernel
def u_psi(q: cudaq.qview):
    h(q[1])


@cudaq.kernel
def u_phi(q: cudaq.qview):
    x(q[0])


@cudaq.kernel
def apply_pauli(q: cudaq.qview):
    x(q[0])
    x(q[1])


@cudaq.kernel
def kernel(num_qubits: int):
    ancilla = cudaq.qubit()
    q = cudaq.qvector(num_qubits)
    h(ancilla)
    cudaq.control(u_phi, ancilla, q)
    cudaq.control(apply_pauli, ancilla, q)
    cudaq.control(u_psi, ancilla, q)


num_qubits = 2
shots = 100000
x_0 = cudaq.spin.x(0)
y_0 = cudaq.spin.y(0)
results = cudaq.observe(kernel, [x_0, y_0], num_qubits, shots_count=shots)
evs = np.array([result.expectation() for result in results])
std_errs = np.sqrt((1 - evs**2) / shots)

print(f"QC result: {evs[0]}+{evs[1]}i ± {std_errs[0]}+{std_errs[1]}i")
print("Numerical result", num_ev)

print(cudaq.__version__)
