import cudaq

cudaq.set_target("nvidia")
                
@cudaq.kernel
def ghz_state(N: int):
    qubits = cudaq.qvector(N)
    h(qubits[0])
    for i in range(N - 1):
        x.ctrl(qubits[i], qubits[i + 1])
    mz(qubits)

counts = cudaq.sample(ghz_state, 5, shots_count=10000)
print(counts)