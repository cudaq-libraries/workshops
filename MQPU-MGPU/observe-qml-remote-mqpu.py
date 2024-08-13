import cudaq
from cudaq import spin
import numpy as np

np.random.seed(1)

backend = 'nvidia-mgpu'
servers = "localhost:30001,localhost:30002"

# Set the target to execute on and query the number of QPUs in the system;
# The number of QPUs is equal to the number of (auto-)launched server instances.
cudaq.set_target("remote-mqpu",
                    backend=backend,
                    auto_launch=str(servers) if servers.isdigit() else "",
                    url="" if servers.isdigit() else servers)
qpu_count = cudaq.get_target().num_qpus()
print("Number of virtual QPUs:", qpu_count)

qubit_count = 30
sample_count = 2

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

# We split our parameters into 2 arrays 
xi = np.split(parameters,2)

print('We have', parameters.shape[0],
      'parameters which we would like to execute')

print('We split this into', len(xi), 'batches of', xi[0].shape[0], ',',
      xi[1].shape[0])

print('Shape after splitting', xi[0].shape)
asyncresults = []


for i in range(len(xi)):
    for j in range(xi[i].shape[0]):
        asyncresults.append(
            cudaq.observe_async(kernel_rx, ham, xi[i][j, :], qpu_id=i))


print('Energies from multi-GPUs')
for result in asyncresults:
    observe_result = result.get()
    got_expectation = observe_result.expectation()
    print(got_expectation)
