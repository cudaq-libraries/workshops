from typing import Callable, List
import cudaq

def test_simple_sampling_qpe():
    """Test that we can build up a set of kernels, compose them, and sample."""

    @cudaq.kernel
    def iqft(qubits: cudaq.qview):
        N = qubits.size()
        for i in range(N // 2):
            swap(qubits[i], qubits[N - i - 1])

        for i in range(N - 1):
            h(qubits[i])
            j = i + 1
            for y in range(i, -1, -1):
                r1.ctrl(-np.pi / 2**(j - y), qubits[j], qubits[y])

        h(qubits[N - 1])

    @cudaq.kernel
    def tGate(qubit: cudaq.qubit):
        t(qubit)

    @cudaq.kernel
    def xGate(qubit: cudaq.qubit):
        x(qubit)

    @cudaq.kernel
    def qpe(nC: int, nQ: int, statePrep: Callable[[cudaq.qubit], None], oracle: Callable[[cudaq.qubit], None]):
        q = cudaq.qvector(nC + nQ)
        countingQubits = q.front(nC)
        stateRegister = q.back()
        statePrep(stateRegister)
        h(countingQubits)
        for i in range(nC):
            for j in range(2**i):
                cudaq.control(oracle, [countingQubits[i]], stateRegister)
        iqft(countingQubits)
        mz(countingQubits)

    print("gpe")
    cudaq.set_random_seed(13)
    counts = cudaq.sample(qpe, 3, 1, xGate, tGate)
    print("gpe2")
    assert len(counts) == 1
    assert '100' in counts
    print("asserted")

    @cudaq.kernel
    def xGateAfterKernel(qubit: cudaq.qubit):
        x(qubit)

    print("x after")
    counts = cudaq.sample(qpe, 3, 1, xGateAfterKernel, tGate)
    assert len(counts) == 1
    assert '100' in counts
    print("done")

print("gogo")
test_simple_sampling_qpe()
