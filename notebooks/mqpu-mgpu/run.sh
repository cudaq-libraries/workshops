#!/bin/bash

#PJM -L jobenv=singularity
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=1:00:00
#PJM -j

module load singularity-ce
singularity exec --nv docker://nvcr.io/nvidia/quantum/cuda-quantum:0.8.0 mpirun -np 4 python ghz.py --target nvidia --target-option mgpu