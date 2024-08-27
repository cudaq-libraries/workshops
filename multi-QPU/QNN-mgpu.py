import cudaq
from cudaq import spin
import numpy as np

import timeit

np.random.seed(1)

cudaq.set_target("nvidia", option="mqpu")
target = cudaq.get_target()
qpu_count = target.num_qpus()
print("Number of QPUs:", qpu_count)

qubit_count = 20
sample_count = 10000

ham = spin.z(0)

parameter_count = 3*qubit_count

# Below we run a circuit for 10000 different input parameters.
parameters = np.random.default_rng(13).uniform(low=0,high=1,size=(sample_count,parameter_count))

print('Parameter shape: ', parameters.shape)

@cudaq.kernel
def qnn(theta:list[float]):
    qubits = cudaq.qvector(qubit_count)

    count=0
    for i in range(qubit_count):
        u3(theta[count], theta[count+1], theta[count+2], qubits[i])
        count+=3
    
    for i in range(qubit_count-1):
        x.ctrl(qubits[i], qubits[i+1])
    
    x.ctrl(qubits[qubit_count-1], qubits[0])

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
        qpu_id = i * 4 // len(xi)
        asyncresults.append(
            cudaq.observe_async(qnn, ham, xi[i][j, :], qpu_id=qpu_id))

#print('Energies from multi-GPUs')
for result in asyncresults:
    observe_result = result.get()
    got_expectation = observe_result.expectation()
    #print(got_expectation)
end_time = timeit.default_timer()
print(f'Elapsed time (s) for multi-QPU with {qpu_count} QPUs is {end_time-start_time}')

