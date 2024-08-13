import cudaq
from cudaq import spin

cudaq.set_target("nvidia")

@cudaq.kernel
def vqe_circuit(theta: list[float]):
    # Allocate a qubit that is initialised to the |0> state.
    qubit = cudaq.qubit()
    # Define gates and the qubits they act upon.
    rx(theta[0], qubit)
    ry(theta[1], qubit)


# Our hamiltonian will be the Z expectation value of our qubit.
hamiltonian = spin.z(0)

# Initial gate parameters which initialize the qubit in the zero state
initial_param = [0.0, 0.0]

cost_values = []
def cost(parameters):
    """Returns the expectation value as our cost."""
    expectation_value = cudaq.observe(vqe_circuit, hamiltonian, parameters).expectation()
    cost_values.append(expectation_value)
    return expectation_value

initial_cost_value = cost(initial_param)
print('Initial cost value: ', initial_cost_value)
print('Initial parameters: ', initial_param)

# Define a CUDA-Q optimizer.
optimizer = cudaq.optimizers.COBYLA()
optimizer.initial_parameters = initial_param

result = optimizer.optimize(dimensions=2, function=cost)

print('Final cost value: ', result[0])
print('Optimized parameters: ', result[1])