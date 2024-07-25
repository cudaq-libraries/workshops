import cudaq
import numpy as np

cudaq.set_target('nvidia')

qubit_num=2

@cudaq.kernel
def U_psi(q:cudaq.qview):
    h(q[1])

@cudaq.kernel
def U_phi(q:cudaq.qview):
    x(q[0])

@cudaq.kernel  
def ham_cir(q:cudaq.qview):
    x(q[0])
    x(q[1])

@cudaq.kernel
def kernel(n:int):
    ancilla=cudaq.qubit()
    q=cudaq.qvector(n)
    h(ancilla)
    cudaq.control(U_phi,ancilla,q)
    cudaq.control(ham_cir,ancilla,q)
    cudaq.control(U_psi,ancilla,q)
    
    h(ancilla)
    
    mz(ancilla)

shots=50000    
count=cudaq.sample(kernel,qubit_num, shots_count=shots)    
print(count)

mean_val=(count['0']-count['1'])/shots
error=np.sqrt(2*count['0']*count['1']/shots)/shots
print('Observable QC: ', mean_val,'+ -', error)
