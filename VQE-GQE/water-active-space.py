# pip install openfermionpyscf
# pip install openfermion
# python water-active-space.py 

import openfermion
import openfermionpyscf
from openfermion.transforms import jordan_wigner, get_fermion_operator

import timeit


import cudaq
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np

# GPU
cudaq.set_target("nvidia", option="fp64")
# CPU
#cudaq.set_target("qpp-cpu")

# 1- Classical pre-processing:
geometry=[('O', (0.1173,0.0,0.0)), ('H', (-0.4691,0.7570,0.0)), ('H', (-0.4691,-0.7570,0.0))]
basis='631g'
multiplicity=1
charge=0
ncore=3
norb_cas, nele_cas = (4,4)

molecule = openfermionpyscf.run_pyscf(openfermion.MolecularData(geometry, basis, multiplicity,charge))

molecular_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=range(ncore), active_indices=range(ncore,ncore+norb_cas))

fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)

spin_ham=cudaq.SpinOperator(qubit_hamiltonian)

# 2- Quantum computing using UCCSD ansatz

electron_count=nele_cas
qubit_count=2*norb_cas

@cudaq.kernel
def kernel(qubit_num:int, electron_num:int, thetas: list[float]):
        qubits = cudaq.qvector(qubit_num)

        for i in range(electron_num):
                x(qubits[i])

        cudaq.kernels.uccsd(qubits, thetas, electron_num, qubit_num)

parameter_count = cudaq.kernels.uccsd_num_parameters(electron_count,qubit_count)

# Define a function to minimize
def cost(theta):

        exp_val = cudaq.observe(kernel, spin_ham, qubit_count, electron_count, theta).expectation()

        return exp_val


exp_vals = []

def callback(xk):
        exp_vals.append(cost(xk))


# Initial variational parameters.
np.random.seed(42)
x0 = np.random.normal(0, 1, parameter_count)

# Use the scipy optimizer to minimize the function of interest
start_time = timeit.default_timer()
result = minimize(cost,x0,method='COBYLA',callback=callback, options={'maxiter': 300})
end_time=timeit.default_timer()

print('UCCSD-VQE energy=  ', result.fun)
print('Total number of qubits: ', qubit_count)
print('Total number of parameters: ', parameter_count)
print('Total number of terms in the spin hamiltonian: ',spin_ham.get_term_count())
print('Total elapsed time (s): ', end_time-start_time)
