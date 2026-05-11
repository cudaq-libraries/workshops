# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # CUDA-Q Introduction[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cudaq-libraries/workshops/blob/main/notebooks/cudaq_introduction.ipynb)

# ## Installation of CUDA-Q
#
# - Visit [CUDA-Q Quick Start](https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html)
# - To explore more, visit [CUDA-Q installation](https://nvidia.github.io/cuda-quantum/latest/using/install/install.html)

# ## Quantum Circuit basics
#
# The purpose of this notebook is to create and execute quantum circuits below.

# Example of Quantum Circuit

from cudaq.qis import *

import cudaq


@cudaq.kernel
def circuit():
    qubits = cudaq.qvector(3)
    h(qubits[0])
    cx(qubits[0], qubits[1])
    cx(qubits[1], qubits[2])


print(cudaq.draw(circuit))

# ### Qubit allocation
#
# - `cudaq.qubit()`: a single quantum bit (2-level) in the discrete quantum memory space.
#
# ```python
# qubit = cudaq.qubit()
# ```
#
# - `cudaq.qvector(n)`: a multi quantum bit ($2^n$ level) in the discrete quantum memory.
#
# ```python
# qubits = cudaq.qvector(n)
# ```
#
#     
# - These are initialized to the |0> computational basis state.
#
# - Owns the quantum memory, therefore it cannot be copied or moved (no-cloning theorem). It can be passed by reference (i.e., references to qubit vectors).

# ### Quantum Operations
#
#
# - `x`: Not gate (Pauli-X gate)
#
# ```python
# q = cudaq.qubit()
# x(q)
# ```
# - `h`: Hadamard gate
#
# ```python
# q = cudaq.qvector(2)
# h(q[0])
# ```
#
# - `x.ctrl(control, target)` or `([control_1, control_2], target)`: CNOT (Controlled-NOT) gate
#
# ```python
# q = cudaq.qvector(3)
# x.ctrl(q[0], q[1])
# cx(q[0], q[1])  # alias of x.ctrl
# ```
#
# - `rx(angle, qubit)`: rotation around x-axis $\operatorname{rx}(\theta) = e^{-i\theta/2}$
# ```python
# q=cudaq.qubit()
# rx(np.pi, q)
# ```
#
# - `adj`: adjoint transformation
# ```python
# q=cudaq.qubit()
# rx(np.pi, q)
# rx.adj(np.pi, q)
# ```
#
# - `mz`: measure qubits in the computational basis
#
# ```python
# q=cudaq.qvector(2)
# h(q[0])
# x.ctrl(q[0], q[1])
# mz(q)
# ```
#
# - `exp_pauli`: exponential of Pauli operators $\operatorname{exp\_pauli}(\theta, P) = e^{i \theta P}$
#
# ```python
# q = cudaq.qvector(2)
# exp_pauli(theta, [q[p], q[r]], "ZZ")
# ```
#
# To learn more about the quantum operations available in CUDA-Q, visit [this page](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/kernels.html).

# ### Quantum kernel
#
# - To differentiate between host and quantum device code, the CUDA-Q programming model defines the concept of a quantum kernel.
#
# - All quantum kernels must be annotated to indicate they are to be compiled for, and executed on, a specified quantum coprocessor.
#
# - Other language bindings may opt to use other language features to enable function annotation or decoration (e.g. a `@cudaq.kernel()` function decorator in Python and `__qpu__` in C++).
#
# - Quantum kernel can take classical data as input.

# ``` python
# @cudaq.kernel()
# def my_first_entry_point_kernel(x : float):
#    ... quantum code ...
#
# @cudaq.kernel()
# def my_second_entry_point_kernel(x : float, params : list[float]):
#    ... quantum code ...
#
# ```

# - `cudaq.qview`: a non-owning reference to a subset of the discrete quantum memory space. It does not own its elements and can therefore be passed by value or reference. (see [this page](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/types.html#quantum-containers))
#
# - Vectors inside the quantum kernel can be only constructed with specified size
#
# ```python
# @cudaq.kernel
# def kernel(n: int):
#
#    # Not Allowed
#    # i = []
#    # i.append(1)
#
#    # Allowed
#    i = [0 for k in range(5)]
#    j = [0 for _ in range(n)]
#    i[2] = 3
#    f = [1., 2., 3.]
#    k = 0
#    pi = 3.1415926
#
# ```
#
# - To learn more about the CUDA-Q quantum kernel, visit [this page](https://github.com/NVIDIA/cuda-quantum/blob/main/docs/sphinx/specification/cudaq/kernels.rst).

# ### Code Examples

# Single qubit example

from cudaq.qis import *

import cudaq


# We begin by defining the `Kernel` that we will construct our
# program with.
@cudaq.kernel()
def first_kernel():
    """
    This is our first CUDA-Q kernel.
    """
    # Next, we can allocate a single qubit to the kernel via `qubit()`.
    qubit = cudaq.qubit()

    # Now we can begin adding instructions to apply to this qubit!
    # Here we'll just add non-parameterized
    # single qubit gate that is supported by CUDA-Q.
    h(qubit)
    x(qubit)
    y(qubit)
    z(qubit)
    t(qubit)
    s(qubit)

    # Next, we add a measurement to the kernel so that we can sample
    # the measurement results on our simulator!
    mz(qubit)


print(cudaq.draw(first_kernel))

# Multi-qubit example

import cudaq


@cudaq.kernel
def second_kernel(num_qubits: int):
    qubits = cudaq.qvector(num_qubits)

    h(qubits[0])
    x.ctrl(qubits[0], qubits[1])
    cx(qubits[0], qubits[2])  # cx is also ok
    x(qubits[0:4])

    mz(qubits)


print(cudaq.draw(second_kernel, 5))

import cudaq


@cudaq.kernel
def bar(num_qubits: int):
    qubits = cudaq.qvector(num_qubits)
    h(qubits[0:3])
    controls = qubits[0:-1]
    target = qubits[-1]

    x.ctrl(controls, target)

print(cudaq.draw(bar, 10))

import cudaq

@cudaq.kernel
def my_state_prep(qubits: cudaq.qview):
    for i in range(qubits.size // 2, qubits.size):
        x(qubits[i])


@cudaq.kernel
def my_algorithm():
    q = cudaq.qvector(10)
    my_state_prep(q)

print(cudaq.draw(my_algorithm))

# <div class="alert alert-block alert-success">
#
# ### Exerciese 1
#
# Now you can make quantum kernels! Let's make the kernel to create the GHZ state for $n$ qubits $\frac{1}{\sqrt{2}}(|00\dots 0\rangle + |11\dots 1\rangle)$!
#
# **Advanced**: Assume that the qubits are connected in one dimension. Let's build a circuit so that the depth of 2-qubit gates is as small as possible.
# <div>

import cudaq

@cudaq.kernel
def ghz(num_qubits: int):
    q = cudaq.qvector(num_qubits)
    # Write your code here

print(cudaq.draw(ghz, 10))


# ## Execute quantum kernels

# ### Function call

# The kernel can be executed by calling a function. If the results need to be output, the return value and its type must be specified

@cudaq.kernel
def bit_flip(flip: bool = True) -> bool:
    qubit = cudaq.qubit()
    if flip:
        x(qubit)
    result = mz(qubit)
    return result


print(bit_flip(False))

# ### cudaq.sample()
#
# Sample a given quantum circuit for a specified number of shots (circuit execution).
#
# This function takes as input a quantum kernel instance followed by the concrete arguments at which the kernel should be invoked.

import cudaq


@cudaq.kernel
def bell(num_qubits: int):
    qubits = cudaq.qvector(num_qubits)

    h(qubits[0])
    x.ctrl(qubits[0], qubits[1])

    mz(qubits)


print(cudaq.draw(bell, 2))
# Sample the state generated by bell
# shots_count: the number of kernel executions. Default is 1000
counts = cudaq.sample(bell, 2, shots_count=10000)

# Print to standard out
print(counts)

# Fine-grained access to the bits and counts
for bits, count in counts.items():
    print(f"Observed {bits}: {count}")

import cudaq


@cudaq.kernel
def third_example(num_qubits: int, theta: list[float]):
    qubit = cudaq.qvector(num_qubits)

    h(qubit)

    for i in range(0, num_qubits // 2):
        ry(theta[i], qubit[i])

    x.ctrl([qubit[0], qubit[1]], qubit[2])  # ccx
    x.ctrl([qubit[0], qubit[1], qubit[2]], qubit[3])  # cccx
    x.ctrl(qubit[0:3], qubit[3])  # cccx using Python slicing syntax

    mz(qubit)


params = [0.15, 1.5]

print(cudaq.draw(third_example, 4, params))

result = cudaq.sample(third_example, 4, params, shots_count=5000)

print("Result: ", result)

print("Most probable bit string: ", result.most_probable())  # Custom dictionary

# <div class="alert alert-block alert-success">
#     
# ### Exercise 2
#
# Let's run your ghz kernel with sampler! Set `shots_count=10000` and the number of qubits is `10`. Do you obtain the expected result?
#
# </div>

# Write your code here!

# ### cudaq.observe()

# - A common task in variational algorithms is the computation of the expected value of a given operator with respect to a parameterized quantum circuit ($\langle H\rangle_\theta = \langle \psi(\theta)\mid H \mid\psi(\theta)\rangle$).
#
# - The `cudaq.observe()` function is provided to enable one to quickly compute this expectation value via execution of the parameterized quantum circuit.
#
# - In the example below, the obervable H is $H= 5.907 \, I - 2.1433 \, X_0X_1 -2.1433\, Y_0 Y_1 + 0.21829 \, Z_0 -6.125\, Z_1$.

# The example here shows a simple use case for the `cudaq.observe``
# function in computing expected values of provided spin operators.

import cudaq
from cudaq import spin

num_qubits = 2


@cudaq.kernel
def init_state(qubits: cudaq.qview):
    n = qubits.size()
    for i in range(n):
        x(qubits[i])


@cudaq.kernel
def observe_example(theta: float):
    qvector = cudaq.qvector(num_qubits)

    init_state(qvector)
    ry(theta, qvector[1])
    x.ctrl(qvector[1], qvector[0])


spin_operator = (
    5.907
    - 2.1433 * spin.x(0) * spin.x(1)
    - 2.1433 * spin.y(0) * spin.y(1)
    + 0.21829 * spin.z(0)
    - 6.125 * spin.z(1)
)

# Pre-computed angle that minimizes the energy expectation of the `spin_operator`.
angle = 0.59

energy = cudaq.observe(observe_example, spin_operator, angle).expectation()
print(f"Energy is {energy}")

# #### Spin operator
#
# CUDA-Q defines convenience functions in `cudaq.spin` namespace that produce the primitive X, Y, and Z Pauli operators on specified qubit indices which can subsequently be used in algebraic expressions to build up more complicated Pauli tensor products and their sums.
#
# $H= 5.907 \, I - 2.1433 \, X_0X_1 -2.1433\, Y_0 Y_1 + 0.21829 \, Z_0 -6.125\, Z_1$
#
# ```python
# spin_operator = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
#     0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
# ```

from cudaq import spin

operator = (
    0.5 * spin.z(0)
    + spin.x(1)
    + spin.y(0)
    + spin.y(0) * spin.y(1)
    + spin.x(0) * spin.y(1) * spin.z(2)
)

# add some more terms
for i in range(2):
    operator += -2.0 * spin.z(i) * spin.z(i + 1)

print(operator)

print("Total number of terms in the spin operator: ", operator.term_count)

# You can construct `SpinOperator` using `from_word` class method.

op = cudaq.SpinOperator.from_word("XXXX")
print(op)

# <div class="alert alert-block alert-success">
#
# #### Exercise 3
#
# Calculate expectation value $\langle \mathrm{ghz} | ZZ...Z | \mathrm{ghz}\rangle$ and $\langle \mathrm{ghz} | XX...X | \mathrm{ghz}\rangle$ for 10 qubits and 20 qubits.
# </div>

# Write your code here!

# ### cudaq.run()
#
# - `cudaq.run()` executes a kernel multiple times and returns the individual return value from each shot.
#
# - Use `cudaq.run()` when the kernel returns values that you want to inspect shot by shot.
#
# - To learn more about cudaq.run(), visit [this page](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/algorithmic_primitives.html#cudaq-run).

import cudaq


@cudaq.kernel
def random_bit() -> bool:
    qubit = cudaq.qubit()
    h(qubit)
    return mz(qubit)


bit_results = cudaq.run(random_bit, shots_count=10)
print(bit_results)
print("Number of True results:", sum(bit_results))

# #### Mid-circuit measurement and conditional execution

from collections import Counter
import cudaq


@cudaq.kernel
def mid_circuit_m(theta: float) -> tuple[bool, bool, bool]:
    qubit = cudaq.qvector(2)
    ancilla = cudaq.qubit()

    ry(theta, ancilla)

    a = mz(ancilla)
    if a:
        x(qubit[0])
        x(ancilla)
    else:
        x(qubit[0])
        x(qubit[1])

    b0 = mz(qubit[0])
    b1 = mz(qubit[1])
    return (a, b0, b1)


def bit_string(*bits: bool) -> str:
    return "".join("1" if bit else "0" for bit in bits)


def sample_like_counts(results: list[tuple[bool, bool, bool]]) -> dict[str, dict[str, int]]:
    global_counts = Counter()
    a_counts = Counter()
    b0_counts = Counter()
    b1_counts = Counter()

    for a, b0, b1 in results:
        global_counts[bit_string(b0, b1, a)] += 1
        a_counts[bit_string(a)] += 1
        b0_counts[bit_string(b0)] += 1
        b1_counts[bit_string(b1)] += 1

    return {
        "__global__": dict(global_counts),
        "a": dict(a_counts),
        "b0": dict(b0_counts),
        "b1": dict(b1_counts),
    }


angle = 0.5

results = cudaq.run(mid_circuit_m, angle, shots_count=10)
print(results)
print(sample_like_counts(results))

# - Here, the first measurement of the ancilla qubit is used for conditional execution.
#
# - `cudaq.run()` returns one tuple per shot, so we reconstruct a sample-like counts dictionary from those shot-by-shot return values.
#
# - The `__global__` key follows qubit allocation order (`b0`, `b1`, then `a`).

### Version information
print(cudaq.__version__)
