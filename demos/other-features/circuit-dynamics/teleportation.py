# python teleportation.py

import cudaq

@cudaq.kernel
def teleport():
    q = cudaq.qvector(3)
    x(q[0])
    h(q[1])

    x.ctrl(q[1], q[2])

    x.ctrl(q[0], q[1])
    h(q[0])

    b0 = mz(q[0])
    b1 = mz(q[1])

    if b1:
        x(q[2])

    if b0:
        z(q[2])

    mz(q[2])

counts = cudaq.sample(teleport, shots_count=100)
print('Altogether: ', counts)
    # Note this is testing that we can provide
    # the register name automatically
b0 = counts.get_register_counts('b0')
print('b0 register: ', b0)
    