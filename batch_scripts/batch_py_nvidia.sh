#!/bin/bash
#BSUB -P TRN024
#BSUB -W 0:05
#BSUB -nnodes 1
#BSUB -alloc_flags gpudefault
#BSUB -J cudaq_py_nv
#BSUB -o cudaq_py_nv_%J.output
#BSUB -e cudaq_py_nv_%J.error

module purge
module use /gpfs/wolf2/olcf/trn024/proj-shared/modulefiles
module load gcc/11.2.0
module load cudaq/0.8.0
module load spectrum-mpi/10.4.0.3-20210112
module -t list

# Here we say give me 1 resource set (n), each
# with a single MPI rank (a), a single core (c), and a
# single GPU (1)
jsrun -n 1 -a 1 -c 1 -g 1 python3 <YOUR_SCRIPT> --target nvidia
