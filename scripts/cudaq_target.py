# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Target Backends
#
# ## Set target
#
# Two options to set the target:
#
# 1. Define the target when running the program:
# ```bash
# python3 program.py [...] --target <target_name>
# ```
#
# 2. Target can be defined in the application code:
# `cudaq.set_target("target_name")` . Then, to run the program, drop the target flag:
# `python3 program.py [...]`
#
#
# ## Target name
#
# 1. State vector simulators:
#     - Single-GPU (Default if an NVIDIA GPU and CUDA runtime libraries are available): `python3 program.py [...] --target nvidia`
#     - Multi-GPUs: `mpirun -np 2 python3 program.py [...] --target nvidia --target-option=mgpu`
# 2. Tensor network simulator:
#     - Single-GPU: `python3 program.py [...] --target tensornet`
#     - Multi-GPUs: `mpirun -np 2 python3 program.py [...] --target tensornet`
# 3. Matrix Product State:
#     - Only supports single-GPU simulation: `python3 program.py [...] --target tensornet-mps`
# 4. NVIDIA Quantum Cloud
#     - Run any of the above backends using NVIDIA-provided cloud GPUs (early access only). To learn more, visit [this page](https://www.nvidia.com/en-us/solutions/quantum-computing/cloud/).
#     - E.g. `cudaq.set_target('nvqc', backend='tensornet')`
# 5. Quantum hardware backend (to learn more, visit [this page](https://nvidia.github.io/cuda-quantum/latest/using/backends/hardware.html)):
#     - ```cudaq.set_target('QPU_name')```. QPU_name could be `ionq`, `quantinuum`, `iqm`, `oqc`, ...etc.
#
#
# To learn more about CUDA-Q backends, visit [this page](https://nvidia.github.io/cuda-quantum/latest/using/backends/backends.html)

import cudaq
from time import perf_counter


@cudaq.kernel
def ghz(n: int):
    q = cudaq.qvector(n)
    h(q[0])

    for i in range(1, n):
        cx(q[0], q[i])


# Run with CPU

cudaq.set_target("qpp-cpu")

counts = cudaq.sample(ghz, 10)
print(counts)

# Statevector simulation tiem scales exponentially.

cudaq.set_target("qpp-cpu")

cpu_times = []
for n in range(10, 22):
    start = perf_counter()
    counts = cudaq.sample(ghz, n)
    end = perf_counter()
    cpu_times.append((n, end - start))

print(cpu_times)

# Simulation with GPU is also exponential, but faster than CPUs for large number of qubits:

# Use GPU
cudaq.set_target("nvidia")

gpu_times = []
for n in range(10, 30):
    start = perf_counter()
    counts = cudaq.sample(ghz, n)
    end = perf_counter()
    gpu_times.append((n, end - start))

print(gpu_times)

# Visualize
import matplotlib.pyplot as plt

plt.plot([i[0] for i in cpu_times], ([i[1] for i in cpu_times]), label="CPU")
plt.plot([i[0] for i in gpu_times], ([i[1] for i in gpu_times]), label="GPU")
plt.xlabel("Number of qubits")
plt.ylabel("Time [sec]")
plt.legend()
plt.savefig("cpugpu.png")

### Version information
print(cudaq.__version__)
