# Single qubit example

import cudaq

# Set the backend target
cudaq.set_target('nvidia')

# We begin by defining the `Kernel` that we will construct our
# program with.
@cudaq.kernel()
def first_kernel():
    '''
    This is our first CUDA-Q kernel.
    '''
    # Next, we can allocate a single qubit to the kernel via `qubit()`.
    qubit = cudaq.qubit()

    # Now we can begin adding instructions to apply to this qubit!
    # Here we'll just add non-parameterized
    # single qubit gate that is supported by CUDA-Q.
    h(qubit)
    x(qubit)
    y(qubit)
    z(qubit)
    s(qubit)    
    t(qubit)

    # Next, we add a measurement to the kernel so that we can sample
    # the measurement results on our simulator!
    mz(qubit)

print(cudaq.draw(first_kernel))