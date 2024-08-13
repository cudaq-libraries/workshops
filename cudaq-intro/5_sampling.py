# Another sampling example

import cudaq

cudaq.set_target('nvidia')

@cudaq.kernel
def sampling_example(N:int, theta:list[float]):
    qubit=cudaq.qvector(N)

    h(qubit)

    for i in range(0,N//2):
        ry(theta[i],qubit[i])
    

    x.ctrl([qubit[0],qubit[1]],qubit[2]) #ccx
    x.ctrl([qubit[0],qubit[1],qubit[2]],qubit[3]) #cccx
    x.ctrl(qubit[0:3],qubit[3]) #cccx using Python slicing syntax

    mz(qubit)

params=[0.15,1.5]

print(cudaq.draw(sampling_example, 4, params))

result=cudaq.sample(sampling_example, 4, params, shots_count=5000)

print('Result: ', result)

print('Most probable bit string: ', result.most_probable())   