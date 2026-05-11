#!/bin/bash

#PBS -q lecture-g
#PBS -W group_list=gt00
#PBS -j oe
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=00:15:00

# Demo jobs:
# qsub notebooks/mqpu-mgpu/run.sh
# qsub -v QUBIT_COUNT=34,CUDAQ_MAX_CPU_MEMORY_GB=NONE notebooks/mqpu-mgpu/run.sh
# qsub -l select=2:mpiprocs=1 -v QUBIT_COUNT=34,TARGET_OPTION=mgpu notebooks/mqpu-mgpu/run.sh
# qsub -l select=2:mpiprocs=1 -v QUBIT_COUNT=35,TARGET_OPTION=mgpu,CUDAQ_MAX_CPU_MEMORY_GB=NONE notebooks/mqpu-mgpu/run.sh

cd "${PBS_O_WORKDIR}"

module load singularity/4.2.1

IMAGE="docker://nvcr.io/nvidia/quantum/cuda-quantum:cu13-0.14.2"
SCRIPT="notebooks/mqpu-mgpu/ghz.py"
SINGULARITY="$(command -v singularity)"
QUBIT_COUNT="${QUBIT_COUNT:-33}"
TARGET_OPTION="${TARGET_OPTION:-}"

export QUBIT_COUNT
export SINGULARITY_CACHEDIR="${PWD}/.singularity-cache"

SIF="${SINGULARITY_CACHEDIR}/cuda-quantum-cu13-0.14.2.sif"
CUDAQ_ARGS=(--target nvidia)
if [ -n "${TARGET_OPTION}" ]; then
  CUDAQ_ARGS+=(--target-option "${TARGET_OPTION}")
fi

mkdir -p "${SINGULARITY_CACHEDIR}"
[ -f "${SIF}" ] || "${SINGULARITY}" pull "${SIF}" "${IMAGE}"

if [ "${TARGET_OPTION}" = "mgpu" ]; then
  export UCX_TLS="tcp,cuda_copy,self"

  MPI_PROCS="$(wc -l < "${PBS_NODEFILE}")"
  MPI_ENV=(-x QUBIT_COUNT -x UCX_TLS)

  if [ -n "${CUDAQ_MAX_CPU_MEMORY_GB:-}" ]; then
    export CUDAQ_MAX_CPU_MEMORY_GB
    MPI_ENV+=(-x CUDAQ_MAX_CPU_MEMORY_GB)
  fi

  unset OMPI_MCA_mca_base_env_list

  mpirun -np "${MPI_PROCS}" \
    "${MPI_ENV[@]}" \
    /usr/bin/env -u OPAL_PREFIX \
    "${SINGULARITY}" exec --nv "${SIF}" \
    python "${SCRIPT}" "${CUDAQ_ARGS[@]}"
else
  "${SINGULARITY}" exec --nv "${SIF}" \
    python "${SCRIPT}" "${CUDAQ_ARGS[@]}"
fi
