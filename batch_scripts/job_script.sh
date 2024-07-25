#BSUB -P TRN024
#BSUB -W 0:10
#BSUB -nnodes 2
#BSUB -alloc_flags gpudefault
#BSUB -J cudaq_py_nvidia
#BSUB -o cudaq_py_nvidia_%J.output
#BSUB -e cudaq_py_nvidia_%J.error

module purge
module use /gpfs/wolf2/olcf/trn024/proj-shared/modulefiles
module load gcc/11.2.0
module load cudaq/0.8.0
module load spectrum-mpi/10.4.0.3-20210112

# 1 rank with 1 GPU
jsrun -n 1 -a 1 -c 1 -g 1 time python3 ../demos/MQPU-MGPU/observe-qml-mnmgpu.py
# 1 rank with 6 GPUs
jsrun -n 1 -a 1 -c 1 -g 6 time python3 ../demos/MQPU-MGPU/observe-qml-mnmgpu.py
# 2 ranks each with 6 GPUs
jsrun -n 2 -a 1 -c 1 -g 6 time python3 ../demos/MQPU-MGPU/observe-qml-mnmgpu.py
