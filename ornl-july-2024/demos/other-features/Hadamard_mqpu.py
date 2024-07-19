import cudaq
import numpy as np

cudaq.set_target("nvidia-mqpu")

qubit_num=2

target = cudaq.get_target()
qpu_count = target.num_qpus()
print("Number of QPUs:", qpu_count)

@cudaq.kernel
def U_psi(q:cudaq.qview, theta:float):
    ry(theta, q[1])

@cudaq.kernel
def U_phi(q:cudaq.qview, theta: float):
    rx(theta, q[0])

@cudaq.kernel  
def ham_cir(q:cudaq.qview):
    x(q[0])
    x(q[1])

@cudaq.kernel
def kernel(n:int, angle:float, theta:float):
    ancilla=cudaq.qubit()
    q=cudaq.qvector(n)
    h(ancilla)
    cudaq.control(U_phi,ancilla,q,theta)
    cudaq.control(ham_cir,ancilla,q)
    cudaq.control(U_psi,ancilla,q, angle)
    
    h(ancilla)
        
    mz(ancilla)
    
shots=50000  
angle=[0.0, 1.5,3.14,0.7]
theta=[0.6, 1.2 ,2.2 ,3.0]

result=[]
for i in range(4):  
    count=cudaq.sample_async(kernel,qubit_num, angle[i], theta[i], shots_count=shots,qpu_id=i)  
    result.append(count)  

mean_val=np.zeros(len(angle))
i=0
for count in result:
    print(i)
    i_result=count.get()
    print(i_result)
    mean_val[i]=(i_result['0']-i_result['1'])/shots
    error=np.sqrt(2*i_result['0']*i_result['1']/shots)/shots
    print('Observable QC: ',  mean_val[i],'+ -', error)
    i+=1

my_mat=np.zeros((2,2),dtype=float)
m=0
for k in range(2):
    for j in range(2):
        my_mat[k,j]=mean_val[m]
        m+=1   

E,V=np.linalg.eigh(my_mat)

print('Compute eigen-values and eigen-vectors')
print('Eigen values: ')
print(E)

print('Eigenvector: ')
print(V)