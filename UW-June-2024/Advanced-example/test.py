import cudaq
from cudaq import spin

cudaq.set_target('nvidia')

ham=-0.106477- 0.0454063*spin.x(0)*spin.x(1)*spin.y(2)*spin.y(3) +0.174073*spin.z(2
                )*spin.z(3)+0.0454063*spin.y(0)*spin.x(1)*spin.x(2)*spin.y(3)


@cudaq.kernel
def kernel_pauli_word(theta: float, var: cudaq.pauli_word):
    q = cudaq.qvector(4)
    x(q[0])
    x(q[1])
    exp_pauli(theta, q, var)

exp_val = cudaq.observe(kernel_pauli_word, ham, 0.11, 'XXXY').expectation()

print('Expectation value: ', exp_val)