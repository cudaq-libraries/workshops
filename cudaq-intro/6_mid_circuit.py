# Mid-circuit measurment example

import cudaq

cudaq.set_target('nvidia')

@cudaq.kernel
def mid_circuit_m(theta:float):
    qubit=cudaq.qvector(2)
    ancilla=cudaq.qubit()

    x(qubit[0])
    
    ry(theta,ancilla)

    aux=mz(ancilla)
    
    if aux:
        x(ancilla)
    else:
        x(qubit[1])
    
    mz(ancilla)
    mz(qubit)

angle=0.5
result=cudaq.sample(mid_circuit_m, angle)
print(result)