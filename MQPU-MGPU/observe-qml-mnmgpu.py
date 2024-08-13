# python observe-qml-multigpu.py

import cudaq
from cudaq import spin
import numpy as np
import timeit

cudaq.mpi.initialize()

np.random.seed(1)

cudaq.set_target("nvidia", option="mqpu")
target = cudaq.get_target()
my_qpu_count = target.num_qpus()
qpu_count = my_qpu_count * cudaq.mpi.num_ranks()
print(f"My rank {cudaq.mpi.rank()} of {cudaq.mpi.num_ranks()}")
print("Number of my QPUs:", my_qpu_count)
print("Number of QPUs total:", qpu_count)

qubit_count = 24
sample_count = 1200

ham = spin.z(0)


# Below we run a circuit for 1200 different sets of input parameters.
parameters = np.random.default_rng(13).uniform(low=0,high=1,size=(sample_count, qubit_count))

print('Parameter shape: ', parameters.shape)

@cudaq.kernel
def kernel_rx(theta:list[float]):
    qubits = cudaq.qvector(qubit_count)

    for i in range(qubit_count):
        rx(theta[i], qubits)

# split per node
split_params = np.split(parameters, cudaq.mpi.num_ranks())
my_rank_params = split_params[cudaq.mpi.rank()]

# Multi-GPU
# We split our parameters into per-GPU arrays
xi = np.split(my_rank_params, my_qpu_count)

print('We have', parameters.shape[0],
      'parameter sets which we would like to execute')

print('We have', my_rank_params.shape[0],
      'parameter sets on this rank')

print('Number of param sets on this rank:', len(xi))
print('Shape of each parameter set after splitting:', xi[0].shape)
asyncresults = []

start_time = timeit.default_timer()

# For each batch
for i in range(len(xi)):
    # For each parameter set
    for j in range(xi[i].shape[0]):
        asyncresults.append(
            cudaq.observe_async(kernel_rx, ham, xi[i][j], qpu_id=i))



exp_list = []
for result in asyncresults:
    observe_result = result.get() # sync happens here.
    got_expectation = observe_result.expectation()
    exp_list.append(got_expectation)

end_time = timeit.default_timer()

print(f'Elapsed time (s) for multi-QPU with {my_qpu_count} QPUs is {end_time-start_time}')
print(f"My rank has {len(exp_list)} results")
total_results = cudaq.mpi.all_gather(len(exp_list)*cudaq.mpi.num_ranks(), exp_list)
print(f"My rank has {len(total_results)} results after all gather")


cudaq.mpi.finalize()
