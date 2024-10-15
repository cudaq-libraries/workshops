import cudaq

cudaq.mpi.initialize()


@cudaq.kernel
def kernel(qubit_count: int):
    # Allocate our qubits.
    qvector = cudaq.qvector(qubit_count)
    # Place the first qubit in the superposition state.
    h(qvector[0])
    # Loop through the allocated qubits and apply controlled-X,
    # or CNOT, operations between them.
    for qubit in range(qubit_count - 1):
        cx(qvector[qubit], qvector[qubit + 1])
    # Measure the qubits.
    mz(qvector)


# print("Preparing GHZ state for", qubit_count, "qubits.")
qubit_count = 33
counts = cudaq.sample(kernel, qubit_count)

if cudaq.mpi.rank() == 0:
    print(counts)

cudaq.mpi.finalize()
