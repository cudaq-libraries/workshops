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

# # CUDA-Q Introduction

# ## Installation of CUDA-Q
#
# - Visit [CUDA-Q Quick Start](https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html)
# - To explore more, visit [CUDA-Q installation](https://nvidia.github.io/cuda-quantum/latest/using/install/install.html)

# ## Quantum Circuit basics
#
# The purpose of this notebook is to create and execute quantum circuits below.

# Example of Quantum Circuit

import numpy as np

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
# - `rx(angle, qubit)`: rotation around x-axis
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
# To learn more about the quantum operations available in CUDA-Q, visit [this page](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/kernels.html).

# Gate examples

@cudaq.kernel
def do_nothing():
    q = cudaq.qubit()

@cudaq.kernel
def x_gate():
    q = cudaq.qubit()
    x(q)

@cudaq.kernel
def h_gate():
    q = cudaq.qubit()
    h(q)

@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])    
    x.ctrl(q[0], q[1])
#    cx(q[0], q[1])


print("initial state:", np.array(cudaq.get_state(do_nothing)))

print("apply X:", np.array(cudaq.get_state(x_gate)))

print("apply H:", np.array(cudaq.get_state(h_gate)))

print("Bell state:", np.array(cudaq.get_state(bell)))

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

# - CUDA-Q kernels can serve as input to other quantum kernels and invoked by kernel function body code.
#
#
# ```python
# @cudaq.kernel()
# def my_state_prep(qubits : cudaq.qview):
#     ... apply state prep operations on qubits ...
#
# @cudaq.kernel()
# def my_generic_algorithm(state_prep : Callable[[cudaq.qview], None]):
#     q = cudaq.qvector(10)
#     state_prep(q)
#     ...
#
# my_generic_algorithm(my_state_prep)
# ```
#
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


# <div class="alert alert-block alert-success">
#
# ### Exerciese 1
#
# Now you can make quantum kernels! Let's make the kernel to create the GHZ state for $n$ qubits $\frac{1}{\sqrt{2}}(|00\dots 0\rangle + |11\dots 1\rangle)$!
#
# **Advanced**: Assume that the qubits are connected in one dimension. Let's build a circuit so that the depth of 2-qubit gates is as small as possible.
# <div>

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

from typing import Callable


@cudaq.kernel()
def my_state_prep(qubits: cudaq.qview):
    for i in range(qubits.size // 2):
        x(qubits[i])


@cudaq.kernel()
def my_generic_algorithm(state_prep: Callable[[cudaq.qview], None]):
    q = cudaq.qvector(10)
    state_prep(q)


print(cudaq.sample(my_generic_algorithm, my_state_prep))

# ###  Mid-circuit measurement & conditional sampling

import cudaq


@cudaq.kernel
def mid_circuit_m(theta: float):
    qubit = cudaq.qvector(2)
    ancilla = cudaq.qubit()

    ry(theta, ancilla)

    aux = mz(ancilla)
    if aux:
        x(qubit[0])
        x(ancilla)
    else:
        x(qubit[0])
        x(qubit[1])

    anc = mz(ancilla)
    sys = mz(qubit)


angle = 0.5

result = cudaq.sample(mid_circuit_m, angle)
print(result)

# - Here, we see that we have measured the ancilla qubit to a register named ```aux```.
#
# - If any measurements appear in the kernel, then only the measured qubits will appear in the ```__global__``` register, and they will be sorted in qubit allocation order.
#
# - To learn more about cudaq.sample(), visit [this page](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/algorithmic_primitives.html#cudaq-sample).

# <div class="alert alert-block alert-success">
#     
# ### Exercise 2
#
# Let's run your ghz kernel with sampler! Set `shots_count=10000`. Do you obtain the expected result?
#
# </div>

# Write your code here!

# ### cudaq.observe()

# - A common task in variational algorithms is the computation of the expected value of a given observable with respect to a parameterized quantum circuit ($\langle H\rangle_\theta = \langle \psi(\theta)\mid H \mid\psi(\theta)\rangle$).
#
# - The `cudaq.observe()` function is provided to enable one to quickly compute this expectation value via execution of the parameterized quantum circuit.
#
# - In the example below, the obervable H is $H= 5.907 \, I - 2.1433 \, X_0X_1 -2.1433\, Y_0 Y_1 + 0.21829 \, Z_0 -6.125\, Z_1$.

# The example here shows a simple use case for the `cudaq.observe``
# function in computing expected values of provided spin hamiltonian operators.

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

# ### Spin Hamiltonian operator
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

hamiltonian = (
    0.5 * spin.z(0)
    + spin.x(1)
    + spin.y(0)
    + spin.y(0) * spin.y(1)
    + spin.x(0) * spin.y(1) * spin.z(2)
)

# add some more terms
for i in range(2):
    hamiltonian += -2.0 * spin.z(i) * spin.z(i + 1)

print(hamiltonian)

print("Total number of terms in the spin hamiltonian: ", hamiltonian.get_term_count())

# ### Parameterized Circuit

import cudaq
from cudaq import spin


@cudaq.kernel
def param_circuit(theta: list[float]):
    # Allocate a qubit that is initialised to the |0> state.
    qubit = cudaq.qubit()
    # Define gates and the qubits they act upon.
    rx(theta[0], qubit)
    ry(theta[1], qubit)


# Our hamiltonian will be the Z expectation value of our qubit.
hamiltonian = spin.z(0)

# Initial gate parameters which initialize the qubit in the zero state
parameters = [0.0, 0.0]

print(cudaq.draw(param_circuit, parameters))

# Compute the expectation value using the initial parameters.
expectation_value = cudaq.observe(param_circuit, hamiltonian, parameters).expectation()

print("Expectation value of the Hamiltonian: ", expectation_value)

# You can construct `SpinOperator` using `from_word` class method.

op = cudaq.SpinOperator.from_word("XXXX")
print(op)

# <div class="alert alert-block alert-success">
#
# ### Exercise 3
#
# Calculate expectation value $\langle \mathrm{ghz} | ZZ...Z | \mathrm{ghz}\rangle$ and $\langle \mathrm{ghz} | XX...X | \mathrm{ghz}\rangle$ for 10 qubits and 20 qubits.
# </div>

# Write your codes here!

# ## Internal Representations
# To look at the MLIR and QIR generated from your code
#
# ### MLIR

import cudaq


@cudaq.kernel
def kernel():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])


# Look at the MLIR
print(kernel)

# ### QIR

# Look at the QIR
print(cudaq.translate(kernel, format="qir"))

# ### OPENQASM 2

print(cudaq.translate(kernel, format="openqasm2"))

### Version information
print(cudaq.__version__)
