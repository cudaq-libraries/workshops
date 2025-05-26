#!/bin/bash

#PBS -q lecture-g
#PBS -W group_list=gt00
#PBS -j oe
#PBS -l select=1:mpiprocs=4

cd ${PBS_O_WORKDIR}

source /work/gt00/share/cudaq-env/miniconda3/bin/activate
source activate cuda-quantum

python3 test.py
