# python observe-qml-1gpu.py 

import cudaq
from cudaq import spin
import numpy as np
import timeit

np.random.seed(1)

cudaq.set_target("nvidia-mqpu")
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
        rx(theta[i], qubits[i])

#single GPU
start_time = timeit.default_timer()

result = cudaq.observe(kernel_rx, ham, parameters)
energies = np.array([r.expectation() for r in result])

end_time = timeit.default_timer()
print('Elapsed time (s) for single GPU: ', end_time-start_time)

#print('Energies from single GPU')
#print(energies)