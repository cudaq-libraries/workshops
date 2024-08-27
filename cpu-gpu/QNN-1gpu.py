# To run the code: python QNN-1gpu.py

import cudaq
from cudaq import spin
import numpy as np

import timeit

np.random.seed(1)

# GPU
cudaq.set_target("nvidia")

#CPU
#cudaq.set_target('qpp-cpu')


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
    
          
# Single-GPU: broadcasting:

start_time = timeit.default_timer()

result = cudaq.observe(qnn, ham, parameters)
energies = np.array([r.expectation() for r in result])

end_time = timeit.default_timer()
print('Elapsed time (s) for single GPU: ', end_time-start_time)
