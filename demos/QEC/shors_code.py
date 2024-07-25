import cudaq
from cudaq import spin

import numpy as np

def shor_kernel() -> cudaq.Kernel:
    kernel = cudaq.make_kernel()
    data_qubits = kernel.qalloc(3*3)

    # X repetition is "outer"
    for i in range(1, 3):
        kernel.cx(data_qubits[0], data_qubits[i*3])

    for i in range(0, 3):
        kernel.h(data_qubits[i*3])

    # Z repetition is "inner"
    for i in range(0, 3):
        ctrl = i*3
    for j in range(1, 3):
        kernel.cx(data_qubits[ctrl], data_qubits[ctrl+j])

    # Logical qubit built, now add an error
    # kernel.z(data_qubits[8])

    # Now measure syndromes
    ancilla_qubits = kernel.qalloc(8)
    # s_0
    kernel.cx(data_qubits[0], ancilla_qubits[0])
    kernel.cx(data_qubits[1], ancilla_qubits[0])
    # s_1
    kernel.cx(data_qubits[0], ancilla_qubits[1])
    kernel.cx(data_qubits[2], ancilla_qubits[1])
    # s_2
    kernel.cx(data_qubits[3], ancilla_qubits[2])
    kernel.cx(data_qubits[4], ancilla_qubits[2])
    # s_3
    kernel.cx(data_qubits[3], ancilla_qubits[3])
    kernel.cx(data_qubits[5], ancilla_qubits[3])
    # s_4
    kernel.cx(data_qubits[6], ancilla_qubits[4])
    kernel.cx(data_qubits[7], ancilla_qubits[4])
    # s_5
    kernel.cx(data_qubits[6], ancilla_qubits[5])
    kernel.cx(data_qubits[8], ancilla_qubits[5])

    # HZH = X
    kernel.h(data_qubits)

    # s_6
    kernel.cx(data_qubits[0], ancilla_qubits[6])
    kernel.cx(data_qubits[1], ancilla_qubits[6])
    kernel.cx(data_qubits[2], ancilla_qubits[6])
    kernel.cx(data_qubits[3], ancilla_qubits[6])
    kernel.cx(data_qubits[4], ancilla_qubits[6])
    kernel.cx(data_qubits[5], ancilla_qubits[6])
    # s_7
    kernel.cx(data_qubits[0], ancilla_qubits[7])
    kernel.cx(data_qubits[1], ancilla_qubits[7])
    kernel.cx(data_qubits[2], ancilla_qubits[7])
    kernel.cx(data_qubits[6], ancilla_qubits[7])
    kernel.cx(data_qubits[7], ancilla_qubits[7])
    kernel.cx(data_qubits[8], ancilla_qubits[7])

    kernel.h(data_qubits)

    kernel.mz(ancilla_qubits)

    return kernel

counts = cudaq.sample(shor_kernel(), shots_count=1000)
# bit_string = counts.most_probable()
# print(bit_string)
counts.dump()


# stabilizers
# {zi zj} {x'i, x'j}
# s_0 = z_0 z_1
# s_1 = z_0 z_2
#
# s_2 = z_3 z_4
# s_3 = z_3 z_5
#
# s_4 = z_6 z_7
# s_5 = z_6 z_8
#
# s_6 = x_0 x_1 x_2 , x_3 x_4 x_5
# s_7 = x_0 x_1 x_2 , x_6 x_7 x_8






