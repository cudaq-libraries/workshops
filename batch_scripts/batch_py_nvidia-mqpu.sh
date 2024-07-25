#!/bin/bash
#BSUB -P TRN024
#BSUB -W 0:05
#BSUB -nnodes 1
#BSUB -alloc_flags gpudefault
#BSUB -J cudaq_py_mqpu
#BSUB -o cudaq_py_mqpu_%J.output
#BSUB -e cudaq_py_mqpu_%J.error

module module purge
module use /gpfs/wolf2/olcf/trn024/proj-shared/modulefiles
module load gcc/11.2.0
module load cudaq/0.8.0
module load spectrum-mpi/10.4.0.3-20210112
module -t list

# Here we say give me 2 resource sets (n), each
# with a single MPI rank (a), a single core (c), and a
# single GPU (1). So this gives us 2 MPI ranks total
# and each one will have a dedicated GPU.
jsrun -n 2 -a 1 -c 1 -g 1 python3 <YOUR_SCRIPT> --target nvidia-mqpu
