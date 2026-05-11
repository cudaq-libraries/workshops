import os

import cudaq


use_mpi = "mgpu" in cudaq.get_target().simulator.lower()
if use_mpi:
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


qubit_count = int(os.environ.get("QUBIT_COUNT", "33"))
counts = cudaq.sample(kernel, qubit_count)

if not use_mpi or cudaq.mpi.rank() == 0:
    print("qubit_count:", qubit_count)
    if "CUDAQ_MAX_CPU_MEMORY_GB" in os.environ:
        print("CUDAQ_MAX_CPU_MEMORY_GB:", os.environ["CUDAQ_MAX_CPU_MEMORY_GB"])
    print(counts)

if use_mpi:
    cudaq.mpi.finalize()
