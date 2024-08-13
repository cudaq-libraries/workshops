# CUDAQ_MGPU_FUSE=4 python c2h4-vqe_24q.py 

import openfermion
import openfermionpyscf
from openfermion.transforms import jordan_wigner, get_fermion_operator

import timeit


import cudaq
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np

# GPU
cudaq.set_target("nvidia", option="mgpu")

# 1- Classical pre-processing:

geometry=[('C',(0.000,0.000,0.6695)), ('C', (0.000,0.000,-0.6695)), \
    ('H',( 0.000,0.9289,1.2321)), ('H', (0.000,-0.9289,1.2321)),\
        ('H', (0.0000, 0.9289, -1.2321)), ('H', (0.000,-0.9289,-1.2321))]
basis='sto3g'
multiplicity=1
charge=0
ncore=2
norb_cas, nele_cas = (12,12)

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

# Initial variational parameters.
np.random.seed(42)
x0 = np.random.normal(0, 1, parameter_count)

start_time = timeit.default_timer()
result=cost(x0)
end_time=timeit.default_timer()

print('Result for single vqe: ', result)
print('Total elapsed time: ', end_time-start_time)
