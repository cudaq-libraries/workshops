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

# # CUDA-Q Internal Representations
#
# This notebook inspects the internal and external forms generated from one CUDA-Q kernel.
#
# CUDA-Q maps quantum kernels to the Quake MLIR dialect. The same kernel can then be lowered to QIR or exported as OpenQASM 2 with `cudaq.translate()`. This notebook is not a full compiler tutorial; it is a quick way to see how the Bell kernel appears at each level.
#
# References:
#
# - [Working with the CUDA-Q IR](https://nvidia.github.io/cuda-quantum/latest/using/extending/cudaq_ir.html)
# - [Quake Dialect](https://nvidia.github.io/cuda-quantum/latest/specification/quake-dialect.html)
# - [cudaq.translate API](https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html#cudaq.translate)
#
# ## MLIR

import cudaq


@cudaq.kernel
def kernel():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])


# Look at the MLIR
print(kernel)

# The important lines in the MLIR output are:
#
# - `quake.alloca !quake.veq<2>` allocates a two-qubit vector.
# - `quake.extract_ref` extracts a reference to one qubit in that vector.
# - `quake.h` applies the Hadamard gate.
# - `quake.x [%control] target` applies a controlled-X operation, equivalent to `cx`.
# - `quake.dealloc` releases the qubit vector.

# ## Translated Formats
#
# `cudaq.translate()` returns a string representation of the kernel. The official Python API lists `qir`, `qir-full`, `qir-base`, `qir-adaptive`, and `openqasm2` as available format names.
#
# - `qir` and its variants are LLVM/QIR forms used closer to the execution toolchain.
# - `openqasm2` is a compact circuit exchange format that is easier to read for small gate-level circuits.
# - OpenQASM 2 translation has limitations for kernels with arguments, so this notebook uses a no-argument kernel.

# ## QIR

# Look at the QIR
print(cudaq.translate(kernel, format="qir"))

# ## OPENQASM 2

print(cudaq.translate(kernel, format="openqasm2"))

### Version information
print(cudaq.__version__)
