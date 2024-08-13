import cudaq
import timeit

cudaq.mpi.initialize()

# qubit_count = 33
qubit_count = 28

@cudaq.kernel
def kernel(qubit_num: int):
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

start = timeit.default_timer()
#print("Preparing GHZ state for", qubit_count, "qubits.")
counts = cudaq.sample(kernel, qubit_count)
end = timeit.default_timer()
if cudaq.mpi.rank() == 0:
    print(counts)
    print("time:", end-start)

cudaq.mpi.finalize()
