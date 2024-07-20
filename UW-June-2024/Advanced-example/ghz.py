import cudaq

cudaq.mpi.initialize()

qubit_num = 30


@cudaq.kernel
def kernel():
    # Allocate our qubits.
    qvector = cudaq.qvector(qubit_num)
    # Place the first qubit in the superposition state.
    h(qvector[0])
    # Loop through the allocated qubits and apply controlled-X,
    # or CNOT, operations between them.
    for qubit in range(qubit_num - 1):
        x.ctrl(qvector[qubit], qvector[qubit + 1])
    # Measure the qubits.
    mz(qvector)


counts = cudaq.sample(kernel)

if cudaq.mpi.rank() == 0:
    print("Preparing GHZ state for", qubit_num, "qubits.")
    print(counts)


cudaq.mpi.finalize()
