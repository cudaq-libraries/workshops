import cudaq

cudaq.set_target('nvidia')

@cudaq.kernel
def mid_circuit_m(theta:float):
    qubit=cudaq.qvector(2)
    ancilla=cudaq.qubit()

    ry(theta,ancilla)

    aux=mz(ancilla)
    if aux:
        x(qubit[0])
        x(ancilla)
    else:
        x(qubit[0])
        x(qubit[1])
    
    mz(ancilla)
    mz(qubit)

angle=0.5
result=cudaq.sample(mid_circuit_m, angle)
print('Sampling result', result)

