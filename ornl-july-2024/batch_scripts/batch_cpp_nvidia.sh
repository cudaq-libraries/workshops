#!/bin/bash
#BSUB -P TRN024
#BSUB -W 0:05
#BSUB -nnodes 1
#BSUB -alloc_flags gpudefault
#BSUB -J cudaq_cpp_nvidia
#BSUB -o cudaq_cpp_nvidia_%J.output
#BSUB -e cudaq_cpp_nvidia_%J.error

module purge
module use /gpfs/wolf2/olcf/trn024/proj-shared/modulefiles
module load gcc/11.2.0
module load cudaq/0.8.0
module load spectrum-mpi/10.4.0.3-20210112
module -t list

# This assumes you have compiled your code with
# --target nvidia

# Here we say give me 1 resource set (n), each
# with a single MPI rank (a), a single core (c), and a
# single GPU (1)
jsrun -n 1 -a 1 -c 1 -g 1 <YOUR_EXEC>
