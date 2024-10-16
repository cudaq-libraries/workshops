#!/usr/bin/bash

#SBATCH --partition=qc-a100
#SBATCH --nodes=1
#SBATCH --time=1:00:00

singularity exec --nv docker://nvcr.io/nvidia/quantum/cuda-quantum:0.8.0 mpirun -np 4 python ghz.py --target nvidia --target-option mgpu
