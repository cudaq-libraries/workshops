import cudaq
from typing import Callable, List

cudaq.set_target('nvidia')

@cudaq.kernel
def mid_circuit_m(theta:float):
    qubit=cudaq.qvector(2)
    ancilla=cudaq.qubit()

    ry(theta,ancilla)

    aux=mz(ancilla)
    if aux:
        x(qubit[0])
        x(ancilla)
    else:
        x(qubit[0])
        x(qubit[1])

    mz(ancilla)
    mz(qubit)

# lq is the 7 data qubit logical qubit register
@cudaq.kernel
def steane_h(lq: cudaq.qview):
    h(lq[0]);
    h(lq[1]);
    h(lq[2]);
    h(lq[3]);
    h(lq[4]);
    h(lq[5]);
    h(lq[6]);


# lq is the 7 data qubit logical qubit register
# x(lq) is valid, but only need final 3 qubits
@cudaq.kernel
def steane_x(lq: cudaq.qview):
    x(lq[4]);
    x(lq[5]);
    x(lq[6]);


# lq is the 7 data qubit logical qubit register
# z(lq) is valid, but only need final 3 qubits
@cudaq.kernel
def steane_z(lq: cudaq.qview):
    z(lq[4]);
    z(lq[5]);
    z(lq[6]);


# lq is the 7 data qubit logical qubit register
@cudaq.kernel
def steane_s(lq: cudaq.qview):
    s.adj(lq[0]);
    s.adj(lq[1]);
    s.adj(lq[2]);
    s.adj(lq[3]);
    s.adj(lq[4]);
    s.adj(lq[5]);
    s.adj(lq[6]);


# lq is the 7 data qubit logical qubit register
@cudaq.kernel
def steane_cx(l_ctrl: cudaq.qview, l_target: cudaq.qview):
    x.ctrl(l_ctrl[0], l_target[0]);
    x.ctrl(l_ctrl[1], l_target[1]);
    x.ctrl(l_ctrl[2], l_target[2]);
    x.ctrl(l_ctrl[3], l_target[3]);
    x.ctrl(l_ctrl[4], l_target[4]);
    x.ctrl(l_ctrl[5], l_target[5]);
    x.ctrl(l_ctrl[6], l_target[6]);


@cudaq.kernel
def steane_prep_logical_zero_flagged(q: cudaq.qview, ancilla: cudaq.qview):
    h(q[0]);
    h(q[4]);
    h(q[6]);
    x.ctrl(q[0], q[1]);
    x.ctrl(q[4], q[5]);
    x.ctrl(q[6], q[3]);
    x.ctrl(q[6], q[5]);
    x.ctrl(q[4], q[2]);
    x.ctrl(q[0], q[3]);
    x.ctrl(q[4], q[1]);
    x.ctrl(q[3], q[2]);

    x.ctrl(q[1], ancilla[0])
    x.ctrl(q[3], ancilla[0])
    x.ctrl(q[5], ancilla[0])

@cudaq.kernel
def steane_prep_logical_zero(q: cudaq.qview):
    h(q[0]);
    h(q[4]);
    h(q[6]);
    x.ctrl(q[0], q[1]);
    x.ctrl(q[4], q[5]);
    x.ctrl(q[6], q[3]);
    x.ctrl(q[6], q[5]);
    x.ctrl(q[4], q[2]);
    x.ctrl(q[0], q[3]);
    x.ctrl(q[4], q[1]);
    x.ctrl(q[3], q[2]);

@cudaq.kernel
def stabilizer_memory(
        num_data_qubits: int,
        num_stabs: int,
        stabs: List[int]):
    num_anc = int(len(stabs)/num_stabs)
    data = cudaq.qvector(num_data_qubits);
    amx = cudaq.qvector(num_anc);
    amz = cudaq.qvector(num_anc);


    steane_prep_logical_zero(data);

    # Starting with fresh qubits here,
    # in general should reset.
    # Now apply x_stabilizer circuit
    h(amx);
    for si in range(num_stabs):
        for xi in range(num_anc):
            di = stabs[si*4 + xi]
            x.ctrl(amx[si], data[di]);
    h(amx);

    # Now apply z_stabilizer circuit
    for si in range(num_stabs):
        for zi in range(num_anc):
            di = stabs[si*4 + zi]
            x.ctrl(data[di], amz[si]);

    x_readout1 = mz(amx);
    z_readout1 = mz(amz);

    data_readout = mz(data);


@cudaq.kernel
def run_steane():
    q = cudaq.qvector(7);
    anc = cudaq.qvector(1);

    steane_prep_logical_zero_flagged(q, anc);
    flag = mz(anc);
    data = mz(q);


angle=0.5
result=cudaq.sample(mid_circuit_m, angle)
print('Sampling result', result)

angle=0.5
print(cudaq.draw(run_steane))
result=cudaq.sample(run_steane)
print('Sampling result', result)

# Stein code
stab_0 = [0,1,2,3]
stab_1 = [1,2,4,5]
stab_2 = [2,3,5,6]
# concatenate
stabs = stab_0 + stab_1 + stab_2
print("stabs:", stabs)

n_data_qubits = 7;
nShots = 1000
num_stabs = 3
print("Steane code results:\n")
print(f"lenxstabs: {len(stabs)}")
print(f"lenxstabs/3: {len(stabs)/3}")
print(cudaq.draw(stabilizer_memory, n_data_qubits, 3, stabs))

result = cudaq.sample(stabilizer_memory, n_data_qubits, 3, stabs)

print("Steane result:", result)

