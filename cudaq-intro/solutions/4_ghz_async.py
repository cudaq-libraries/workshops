import cudaq
from cudaq import spin

cudaq.set_target("nvidia")
                
@cudaq.kernel
def ghz_state(N: int):
    qubits = cudaq.qvector(N)
    h(qubits[0])
    for i in range(N - 1):
        x.ctrl(qubits[i], qubits[i + 1])

nqubits = 5

Hx = spin.x(0)
for i in range(1, nqubits):
    Hx = Hx*spin.x(i)

Hz = spin.z(0)
for i in range(1, nqubits):
    Hz = Hz*spin.z(i)


Hx_async_result = cudaq.observe_async(ghz_state, Hx, nqubits)
Hz_async_result = cudaq.observe_async(ghz_state, Hz, nqubits)

print("Expectation value of Hx:", Hx_async_result.get().expectation())
print("Expectation value of Hz:", Hz_async_result.get().expectation())