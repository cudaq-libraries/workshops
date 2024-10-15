#!/bin/bash

#PJM -L jobenv=singularity
#PJM -L rscgrp=cx-debug
#PJM -L node=1
#PJM -j

module load singularity
singularity exec --nv docker://nvcr.io/nvidia/quantum/cuda-quantum:0.8.0 mpirun -np 4 python ghz.py --target nvidia --target-option mgpu