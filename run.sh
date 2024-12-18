#!/bin/bash

#PJM -g gt00
#PJM -L rscgrp=tutorial-a
#PJM -L gpu=1
#PJM -L elapse=00:05:00

module load cudaq
source $MINICONDA_DIR/etc/profile.d/conda.sh
conda activate cudaq-env

python scripts/cudaq_introduction.py
# python scripts/cudaq_target.py
# python scripts/hadamard_test.py
