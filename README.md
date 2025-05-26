**CUDA-Q Hands-on Workshop Materials**
=====================================

Welcome to the CUDA-Q Hands-on Workshop Materials repository! This repository is dedicated to collecting and organizing CUDA-Q code tutorials and materials from the many workshops given to various institutions across the world. The goal is to provide a comprehensive resource for learners and professionals alike, covering a range of beginner to advanced use cases.

Happy learning with CUDA-Q!

### Branch

Materials for each materials are at the branch.
For exapmle, cloning can be done by specifying a branch as follows:
```sh
git clone -b 202505-utokyo --single-branch https://github.com/cudaq-libraries/workshops.git
```

### Environment setting

```sh
module load miniforge3

conda init bash
conda config --add envs_dirs /work/gt00/$USER/.conda/envs
conda config --add pkgs_dirs /work/gt00/$USER/.conda/pkgs

# https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html#install-cuda-q
cuda_version=12.4.0
conda create -y -n cudaq-env python=3.11 pip
conda install -y -n cudaq-env -c "nvidia/label/cuda-${cuda_version}" cuda
conda install -y -n cudaq-env -c conda-forge mpi4py openmpi">=5.0.3" cxx-compiler
conda env config vars set -n cudaq-env LD_LIBRARY_PATH="$CONDA_PREFIX/envs/cudaq-env/lib:$LD_LIBRARY_PATH"
conda env config vars set -n cudaq-env MPI_PATH=$CONDA_PREFIX/envs/cudaq-env
# module load ?
# source $CONDA_PREFIX/lib/python3.11/site-packages/distributed_interfaces/activate_custom_mpi.sh
```
