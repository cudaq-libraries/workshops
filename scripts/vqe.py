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

# # Variational Quantum Eigensolver

# A common application of the Variational Quantum Eigensolver (VQE) algorithm is to compute the ground state energy of a molecular system. The code below demonstrates how to perform classical preprocessing for a $H_2$ molecule (i.e. obtain the integrals from a Hartree-Fock computation to build the molecular Hamiltonian), prepare the initial Hartree-Fock state on the quantum register, add the parameterized UCCSD ansatz to the kernel, and select the COBYLA optimizer.  We are then ready to call `cudaq:vqe` to estimate the minimum energy of the system.

# pip install openfermionpyscf

import cudaq

cudaq.set_target("nvidia")

# The problem of interest here is a chain of hydrogen atoms seperated along the z-axis at a fixed interval called the bond distance.
#
# The interatomic electrostatic forces due to the electrons and protons and the shielding by the neutrons creates a chemical system whose energy can be minimised to find a stable configuration.
#
# Let us first begin by defining the molecule and other metadata about the problem.
#

# Number of hydrogen atoms.
hydrogen_count = 2

# Distance between the atoms in Angstroms.
bond_distance = 0.7474

# Define a linear chain of Hydrogen atoms
geometry = [("H", (0, 0, i * bond_distance)) for i in range(hydrogen_count)]

molecule, data = cudaq.chemistry.create_molecular_hamiltonian(geometry, "sto-3g", 1, 0)

electron_count = data.n_electrons
num_qubits = 2 * data.n_orbitals

## Classical solution
import numpy as np

matrix = np.array(molecule.to_matrix())
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

optimizer = cudaq.optimizers.COBYLA()

energy, parameters = cudaq.vqe(
    kernel, molecule, optimizer, parameter_count=parameter_count
)

print(energy)

# ### Integration with Third-Party Optimizers
#
# We can also integrate popular libraries like scipy with CUDA-Q.

from scipy.optimize import minimize


# Define a function to minimize
def cost(theta):

    exp_val = cudaq.observe(kernel, molecule, theta).expectation()

    return exp_val


exp_vals = []


def callback(xk):
    exp_vals.append(cost(xk))


# Initial variational parameters.
np.random.seed(42)
x0 = np.random.normal(0, np.pi, parameter_count)

# Use the scipy optimizer to minimize the function of interest
result = minimize(cost, x0, method="COBYLA", callback=callback, options={"maxiter": 40})

import matplotlib.pyplot as plt

plt.plot(exp_vals)
plt.xlabel("Epochs")
plt.ylabel("Energy")
plt.title("VQE")
plt.savefig("vqe.png")

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
