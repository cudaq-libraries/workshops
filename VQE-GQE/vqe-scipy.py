import cudaq
from cudaq import spin
import scipy

cudaq.set_target("nvidia")

@cudaq.kernel
def vqe_circuit_scipy(theta: list[float]):
    # Allocate a qubit that is initialised to the |0> state.
    qubit = cudaq.qubit()
    # Define gates and the qubits they act upon.
    rx(theta[0], qubit)
    ry(theta[1], qubit)

# Our hamiltonian will be the Z expectation value of our qubit.
hamiltonian = spin.z(0)

# Initial gate parameters which initialize the qubit in the zero state
initial_parameters = [0.0, 0.0]

def cost_scipy(parameters):
    """Returns the expectation value as our cost."""
    expectation_value = cudaq.observe(vqe_circuit_scipy, hamiltonian, parameters).expectation()
    return expectation_value

initial_cost_value = cost_scipy(initial_parameters)
print('Initial cost value: ', initial_cost_value)
print('Initial parameters: ', initial_parameters)

result = scipy.optimize.minimize(cost_scipy,initial_parameters,method='COBYLA')

print('Final cost value: ', result.fun)
print('Optimized parameters: ', result.x)