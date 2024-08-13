# Multi-control gates example
import cudaq

cudaq.set_target('nvidia')

@cudaq.kernel
def bar(N:int):
    qubits=cudaq.qvector(N)
    # front and back: return a direct refernce 
    controls = qubits.front(N - 1)
    target = qubits.back()
    x.ctrl(controls, target)


print(cudaq.draw(bar,4))