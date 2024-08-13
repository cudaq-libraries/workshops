#BSUB -P TRN024
#BSUB -W 0:10
#BSUB -nnodes 1
#BSUB -alloc_flags gpudefault
#BSUB -J cudaq_py_nvidia
#BSUB -o cudaq_py_nvidia_%J.output
#BSUB -e cudaq_py_nvidia_%J.error

module purge
module use /gpfs/wolf2/olcf/trn024/proj-shared/modulefiles
module load gcc/11.2.0
module load cudaq/0.8.0
module load spectrum-mpi/10.4.0.3-20210112
export CUDAQ_MGPU_LIB_MPI=libmpi_ibm.so
# 1 rank with 1 GPU
export CUDAQ_MGPU_FUSE=1
jsrun -n 1 -a 1 -c 1 -g 1 python3 mgpu_ghz.py --target nvidia --target-option mgpu
export CUDAQ_MGPU_FUSE=2
jsrun -n 1 -a 1 -c 1 -g 1 python3 mgpu_ghz.py --target nvidia --target-option mgpu
export CUDAQ_MGPU_FUSE=3
jsrun -n 1 -a 1 -c 1 -g 1 python3 mgpu_ghz.py --target nvidia --target-option mgpu
export CUDAQ_MGPU_FUSE=4
jsrun -n 1 -a 1 -c 1 -g 1 python3 mgpu_ghz.py --target nvidia --target-option mgpu
export CUDAQ_MGPU_FUSE=5
jsrun -n 1 -a 1 -c 1 -g 1 python3 mgpu_ghz.py --target nvidia --target-option mgpu
