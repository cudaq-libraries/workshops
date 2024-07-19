#!/bin/bash
#BSUB -P TRN024
#BSUB -W 0:05
#BSUB -nnodes 1
#BSUB -alloc_flags gpudefault
#BSUB -J cudaq_cpp_nvidia-mqpu
#BSUB -o cudaq_cpp_nvidia-mqpu_%J.output
#BSUB -e cudaq_cpp_nvidia-mqpu_%J.error

module purge
module use /gpfs/wolf2/olcf/trn024/proj-shared/modulefiles
module load gcc/11.2.0
module load cudaq/0.8.0
module load spectrum-mpi/10.4.0.3-20210112
module -t list

# This assumes you have compiled your code with
# --target nvidia-mqpu

# Here we say give me 2 resource sets (n), each
# with a single MPI rank (a), a single core (c), and a
# single GPU (1). So this gives us 2 MPI ranks total
# and each one will have a dedicated GPU.
jsrun -n 2 -a 1 -c 1 -g 1 <YOUR_EXEC>
