# Large scale simulation with Host CPU memory utilization
# Note: This code also works in notebooks. 
# However, I’ve made it a standalone file because environment variables must be set before `import cudaq`

import os
os.environ["CUDAQ_MAX_CPU_MEMORY_GB"] = "None"

import cudaq
import numpy as np

cudaq.set_target("nvidia")


secret_string_length = 35

random_generator = np.random.default_rng(seed=15)
secret_string = random_generator.integers(
    2, size=secret_string_length
)  # Change the secret string to whatever you prefer

@cudaq.kernel
def oracle(register: cudaq.qview, auxiliary_qubit: cudaq.qubit, secret_string: list[int]):

    for index, bit in enumerate(secret_string):
        if bit == 1:
            x.ctrl(register[index], auxiliary_qubit)

@cudaq.kernel
def bernstein_vazirani(secret_string: list[int]):

    qubits = cudaq.qvector(len(secret_string))  # register of size n
    auxiliary_qubit = cudaq.qubit()  # auxiliary qubit

    # Prepare the auxillary qubit.
    x(auxiliary_qubit)
    h(auxiliary_qubit)

    # Place the rest of the register in a superposition state.
    h(qubits)

    # Query the oracle.
    oracle(qubits, auxiliary_qubit, secret_string)

    # Apply another set of Hadamards to the register.
    h(qubits)

    mz(qubits)  # measures only the main register


print(cudaq.draw(bernstein_vazirani, secret_string))

print("Your secret string is", secret_string)
assert secret_string_length == len(secret_string)

print(cudaq.draw(bernstein_vazirani, secret_string))
result = cudaq.sample(bernstein_vazirani, secret_string)
print("Sample result:", result)

print(f"secret bitstring = {secret_string}")
print(f"measured state = {result.most_probable()}")
is_success = "".join(str(i) for i in secret_string) == result.most_probable()
print(f"Were we successful?", is_success)