#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=00:10:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load singularitypro

singularity exec --nv cuda-quantum_0.8.0.sif python ghz.py --target nvidia --target-option mgpu,fp32
