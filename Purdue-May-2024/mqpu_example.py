import cudaq

cudaq.set_target("nvidia-mqpu")

target = cudaq.get_target()
qpu_count = target.num_qpus()
print("Number of QPUs:", qpu_count)


@cudaq.kernel
def mqpu_example(qubit_count: int):
    qubits = cudaq.qvector(qubit_count)
    # Place qubits in superposition state.
    h(qubits)
    # Measure.
    mz(qubits)


count_futures = []
for qpu in range(qpu_count):
    count_futures.append(cudaq.sample_async(mqpu_example, 5, qpu_id=qpu))

for counts in count_futures:
    print(counts.get())