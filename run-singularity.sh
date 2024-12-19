#!/bin/bash

#PJM -g gt00
###PJM -L rscgrp=tutorial-a
#PJM -L rscgrp=lecture-a
#PJM -L gpu=4
#PJM -L elapse=00:05:00
#PJM -L jobenv=singularity

cd $PJM_O_WORKDIR
module load singularity

SIF=$(pwd)/cuda-quantum_cu12-0.9.1.sif
OPT="--nv --bind $(pwd):/home/$PJM_O_LOGNAME"

# VQE
# singularity exec $OPT $SIF pip install --user openfermionpyscf
# singularity exec $OPT $SIF python3 scripts/vqe.py

# MQPU, MGPU
singularity exec $OPT $SIF python3 scripts/mqpu-mgpu/mqpu-mgpu.py --target nvidia --target-option mqpu
singularity exec $OPT $SIF mpirun -np 4 python3 notebooks/mqpu-mgpu/ghz.py --target nvidia --target-option mgpu
