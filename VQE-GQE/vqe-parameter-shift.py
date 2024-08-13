import openfermion
import openfermionpyscf
from openfermion.transforms import jordan_wigner, get_fermion_operator

import timeit

import cudaq
from scipy.optimize import minimize
import numpy as np

# GPU
cudaq.set_target("nvidia", option="fp64")

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

gradient = cudaq.gradients.ParameterShift()

def objective_function(parameter_vector: list[float], \
                       gradient=gradient, hamiltonian=spin_ham, kernel=kernel):


    get_result = lambda parameter_vector: cudaq.observe\
        (kernel, hamiltonian, qubit_count, electron_count, parameter_vector).expectation()
    
    cost = get_result(parameter_vector)
    gradient_vector = gradient.compute(parameter_vector, get_result,cost)
    
    return cost, gradient_vector

# Initial variational parameters.
np.random.seed(42)
init_params = np.random.normal(0, 1, parameter_count)
result_vqe=minimize(objective_function,init_params, method='L-BFGS-B', jac=True, 
                                   tol=1e-8)

print('VQE-UCCSD energy= ', result_vqe.fun)