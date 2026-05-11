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

# # Variational Quantum Eigensolver [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cudaq-libraries/workshops/blob/main/notebooks/vqe.ipynb)

# A common application of the Variational Quantum Eigensolver (VQE) algorithm is to compute the ground state energy of a molecular system. The code below demonstrates how to perform classical preprocessing for a $H_2$ molecule (i.e. obtain the integrals from a Hartree-Fock computation to build the molecular Hamiltonian), prepare the initial Hartree-Fock state on the quantum register, add the parameterized UCCSD ansatz to the kernel, and select the COBYLA optimizer.  We are then ready to call `cudaq:vqe` to estimate the minimum energy of the system.

#pip install openfermionpyscf

import cudaq

cudaq.set_target("nvidia", option="fp64")

# The problem of interest here is a chain of hydrogen atoms seperated along the z-axis at a fixed interval called the bond distance.
#
# The interatomic electrostatic forces due to the electrons and protons and the shielding by the neutrons creates a chemical system whose energy can be minimised to find a stable configuration.
#
# Let us first begin by defining the molecule and other metadata about the problem.
#

import cudaq
import openfermion
import openfermionpyscf
from openfermion.transforms import jordan_wigner, get_fermion_operator

# Number of hydrogen atoms.
hydrogen_count = 2

# Distance between the atoms in Angstroms.
bond_distance = 0.7474

# Define a linear chain of Hydrogen atoms
geometry = [("H", (0, 0, i * bond_distance)) for i in range(hydrogen_count)]

basis = "sto3g"
multiplicity = 1
charge = 0

molecule = openfermionpyscf.run_pyscf(openfermion.MolecularData(geometry, basis, multiplicity, charge))
molecular_hamiltonian = molecule.get_molecular_hamiltonian()
fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
qubit_hamiltonian.compress()

spin_operator = cudaq.SpinOperator(qubit_hamiltonian)

num_qubits = spin_operator.qubit_count
electron_count = molecule.n_electrons

## Classical solution
import numpy as np

matrix = np.array(spin_operator.to_matrix())
print("Ground state energy:", np.linalg.eigvalsh(matrix)[0])


# We now generate a Unitary Coupled-Cluster Singles and Doubles (UCCSD) ansatz from the template provided by CUDA-Q.  

@cudaq.kernel
def kernel(thetas: list[float]):

    qubits = cudaq.qvector(num_qubits)

    for i in range(electron_count):
        x(qubits[i])

    cudaq.kernels.uccsd(qubits, thetas, electron_count, num_qubits)


parameter_count = cudaq.kernels.uccsd_num_parameters(electron_count, num_qubits)

parameter_count


# ### Using CUDA-Q Optimizers
#
# We use the builtin optimizers within CUDA-Q for the minimization procedure. 
#
# Reference: https://nvidia.github.io/cuda-quantum/latest/examples/python/optimizers_gradients.html#1.-Built-in-CUDA-Q-Optimizers-and-Gradients

# Define the cost function to minimize.
def cost(theta):
    exp_val = cudaq.observe(kernel, spin_operator, theta).expectation()
    return exp_val


optimizer = cudaq.optimizers.COBYLA()
optimizer.max_iterations = 40

energy, parameters = optimizer.optimize(parameter_count, cost)

print(energy)

# ### Integration with Third-Party Optimizers
#
# We can also integrate popular libraries like scipy with CUDA-Q.

from scipy.optimize import minimize


exp_vals = []


def callback(xk):
    exp_vals.append(cost(xk))


# Initial variational parameters.
np.random.seed(42)
x0 = np.random.normal(0, np.pi, parameter_count)

# Use the scipy optimizer to minimize the function of interest
result = minimize(cost, x0, method="COBYLA", callback=callback, options={"maxiter": 100})

import matplotlib.pyplot as plt

plt.plot(exp_vals)
plt.xlabel("Epochs")
plt.ylabel("Energy")
plt.title("VQE")
plt.show()

print("Ground state energy:", np.linalg.eigvalsh(matrix)[0])
print("Final energy", exp_vals[-1])

# <div class="alert alert-block alert-success">
#
# ## Exercise
#
# Let’s try increasing the number of hydrogen atoms to more than two, like `4` or `6`. Run the variational quantum eigensolver and try other optimizers and/or ansatz.
# </div>

# Write your codes here

print(cudaq.__version__)
