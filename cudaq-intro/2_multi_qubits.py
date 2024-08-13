# Multi-qubit example

import cudaq

cudaq.set_target('nvidia')

@cudaq.kernel
def second_kernel(N:int):
    qubits=cudaq.qvector(N)

    h(qubits[0])
    
    for i in range(1, N):
        x.ctrl(qubits[0],qubits[i])
        
    z(qubits)

    mz(qubits)

print(cudaq.draw(second_kernel,3))