# python Q-RBM.py 

import cudaq

cudaq.set_target("nvidia-mqpu")

target = cudaq.get_target()
qpu_count = target.num_qpus()
print("Number of QPUs:", qpu_count)

@cudaq.kernel
def qrbm(v_nodes:int, h_nodes:int, ancilla:int, theta: list[float], coupling: list[float]):

    qubits_num=v_nodes+h_nodes+ancilla
    qubits=cudaq.qvector(qubits_num)

    for i in range(v_nodes+h_nodes):
        ry(theta[i],qubits[i])


    a_target=v_nodes+h_nodes
    count=0
    for v in range(v_nodes):
        for h in range(v_nodes,v_nodes+h_nodes):
            ry.ctrl(coupling[count],qubits[v],qubits[h],qubits[a_target])
            x(qubits[v])
            ry.ctrl(coupling[count+1],qubits[v],qubits[h],qubits[a_target])
            x(qubits[v])
            x(qubits[h])
            ry.ctrl(coupling[count+1],qubits[v],qubits[h],qubits[a_target])
            x(qubits[v])
            ry.ctrl(coupling[count],qubits[v],qubits[h],qubits[a_target])
            x(qubits[v])
            x(qubits[h])

            count+=2
            a_target+=1

    mz(qubits)    
    
v_nodes=2
h_nodes=2
ancilla=4

# Initialize the parameters for the RBM
theta=[2.0482, 1.4329, 2.1774, 2.7122]
coupling=[1.8256, 3.1415, 1.8257, 3.1415, 3.1415, 0.4152, 3.1415, 0.9654]

count_futures = []

for qpu in range(3):
    count_futures.append(cudaq.sample_async(qrbm,v_nodes, h_nodes, ancilla, theta, coupling, shots_count=10000,qpu_id=qpu))

for counts in count_futures:
    print(counts.get())
