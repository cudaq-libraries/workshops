# python observe-qml-multigpu.py

import cudaq
from cudaq import spin
import numpy as np
import timeit

np.random.seed(1)

cudaq.set_target("nvidia", option="mqpu")
target = cudaq.get_target()
qpu_count = target.num_qpus()
print("Number of QPUs:", qpu_count)

qubit_count = 10
sample_count = 500

ham = spin.z(0)

parameter_count = qubit_count

# Below we run a circuit for 500 different input parameters.
parameters = np.random.default_rng(13).uniform(low=0,high=1,size=(sample_count,parameter_count))

print('Parameter shape: ', parameters.shape)

@cudaq.kernel
def kernel_rx(theta:list[float]):
    qubits = cudaq.qvector(qubit_count)

    for i in range(qubit_count):
        rx(theta[i], qubits)

# Multi-GPU

# We split our parameters into 4 arrays since we have 4 GPUs available.
xi = np.split(parameters,4)

print('We have', parameters.shape[0],
      'parameters which we would like to execute')

print('We split this into', len(xi), 'batches of', xi[0].shape[0], ',',
      xi[1].shape[0], ',', xi[2].shape[0], ',', xi[3].shape[0])

print('Shape after splitting', xi[0].shape)
asyncresults = []

start_time = timeit.default_timer()

for i in range(len(xi)):
    for j in range(xi[i].shape[0]):
        asyncresults.append(
            cudaq.observe_async(kernel_rx, ham, xi[i][j, :], qpu_id=i))


#print('Energies from multi-GPUs')
for result in asyncresults:
    observe_result = result.get()
    got_expectation = observe_result.expectation()
    # print(got_expectation)

end_time = timeit.default_timer()
print(f'Elapsed time (s) for multi-QPU with {qpu_count} QPUs is {end_time-start_time}')
