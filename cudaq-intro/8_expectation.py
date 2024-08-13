# Expectation value example

# The example here shows a simple use case for the `cudaq.observe``
# function in computing expected values of provided spin hamiltonian operators.

import cudaq
from cudaq import spin

cudaq.set_target('nvidia')

qubit_num=2

@cudaq.kernel
def init_state(qubits:cudaq.qview):
    n=qubits.size()
    for i in range(n):
        x(qubits[i])

@cudaq.kernel
def observe_example(theta: float):
    qvector = cudaq.qvector(qubit_num)

    init_state(qvector)
    ry(theta, qvector[1])
    x.ctrl(qvector[1], qvector[0])


spin_operator = -5.907 + 2.1433 * spin.x(0) * spin.x(1) + 2.1433 * spin.y(
    0) * spin.y(1) - .21829 * spin.z(0) + 6.125 * spin.z(1)

# Pre-computed angle that minimizes the energy expectation of the `spin_operator`.
angle = 0.59

energy = cudaq.observe(observe_example, spin_operator, angle).expectation()
print(f"Energy is {energy}")