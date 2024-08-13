# Asynchronous sampling example

import cudaq

cudaq.set_target('nvidia')

@cudaq.kernel
def asynchronous_example(N:int, theta:list[float]):
    qubit=cudaq.qvector(N)

    h(qubit)

    for i in range(0,N//2):
        ry(theta[i],qubit[i])
    

    x.ctrl([qubit[0],qubit[1]],qubit[2]) #ccx
    x.ctrl([qubit[0],qubit[1],qubit[2]],qubit[3]) #cccx
    x.ctrl(qubit[0:3],qubit[3]) #cccx using Python slicing syntax

    mz(qubit)

params=[0.15,1.5]

print(cudaq.draw(asynchronous_example, 4, params))

async_result=cudaq.sample_async(asynchronous_example, 4, params, shots_count=5000)
print("Sampling triggered on the GPU...")

# In the mean time, let us do calculations on the CPU
print("Doing some CPU work...")
x = 0
for i in range(1000):
    x+=i
print("CPU work is done!")

print("Waiting for sampling result...")
# Now let's check the result
result = async_result.get()

print('Result: ', result)

print('Most probable bit string: ', result.most_probable())   