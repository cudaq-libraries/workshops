#!/bin/bash

#PBS -q lecture-g
#PBS -W group_list=gt00
#PBS -j oe
#PBS -l select=1
#PBS -l walltime=00:15:00

# Usage:
# qsub run.sh
# qsub -v SCRIPT=scripts/vqe.py run.sh

cd "${PBS_O_WORKDIR}"

export MPLBACKEND=Agg

module load singularity/4.2.1

SCRIPT="${SCRIPT:-scripts/cudaq_introduction.py}"
CUDAQ_IMAGE="${CUDAQ_IMAGE:-docker://nvcr.io/nvidia/quantum/cuda-quantum:cu13-0.14.2}"
SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/work/gt00/share/.singularity-cache}"

export PYTHONUSERBASE="${PWD}/.python-userbase"
export PIP_CACHE_DIR="${PWD}/.pip-cache"
export SINGULARITY_CACHEDIR

run_in_container() {
  singularity exec --nv "${CUDAQ_IMAGE}" "$@"
}

if [ ! -f "${SCRIPT}" ]; then
  echo "Script not found: ${SCRIPT}" >&2
  exit 2
fi

mkdir -p "${PYTHONUSERBASE}" "${PIP_CACHE_DIR}" "${SINGULARITY_CACHEDIR}"

echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "Script: ${SCRIPT}"
echo "Image: ${CUDAQ_IMAGE}"
singularity --version

run_in_container python -m pip install --user -e .

run_in_container python --version
run_in_container python -c 'import cudaq; print(cudaq.__version__)'

run_in_container python "${SCRIPT}" ${SCRIPT_ARGS:-}
