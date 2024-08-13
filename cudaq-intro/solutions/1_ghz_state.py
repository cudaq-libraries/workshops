import cudaq

cudaq.set_target("nvidia")
                
@cudaq.kernel
def ghz_state(N: int):
    qubits = cudaq.qvector(N)
    h(qubits[0])
    for i in range(N - 1):
        x.ctrl(qubits[i], qubits[i + 1])
    mz(qubits)

print(cudaq.draw(ghz_state, 5))