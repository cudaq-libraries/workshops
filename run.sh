#!/bin/bash

#PBS -q lecture-g
#PBS -W group_list=gt00
#PBS -j oe
#PBS -l select=1
#PBS -l walltime=00:15:00

set -euo pipefail

cd "${PBS_O_WORKDIR}"

export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
export MPLBACKEND=Agg

module load singularity/4.2.1

: "${SCRIPT:=scripts/cudaq_introduction.py}"
: "${PYTHON:=python}"
: "${INSTALL_PROJECT_DEPS:=1}"
: "${PYTHONUSERBASE:=${PBS_O_WORKDIR}/.python-userbase}"
: "${PIP_CACHE_DIR:=${PBS_O_WORKDIR}/.pip-cache}"
: "${CUDAQ_IMAGE:=docker://nvcr.io/nvidia/quantum/cuda-quantum:cu13-0.14.2}"
: "${SINGULARITY_CACHEDIR:=${PBS_O_WORKDIR}/.singularity-cache}"
: "${SINGULARITY_BINDPATH:=${PBS_O_WORKDIR}:${PBS_O_WORKDIR}}"

export PYTHONUSERBASE
export PIP_CACHE_DIR
export SINGULARITY_CACHEDIR
export SINGULARITY_BINDPATH

if [ ! -f "${SCRIPT}" ]; then
  echo "Script not found: ${SCRIPT}" >&2
  exit 2
fi

mkdir -p "${SINGULARITY_CACHEDIR}"

echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "Script: ${SCRIPT}"
echo "Image: ${CUDAQ_IMAGE}"
echo "Install project dependencies: ${INSTALL_PROJECT_DEPS}"
singularity --version

if [ "${INSTALL_PROJECT_DEPS}" = "1" ]; then
  (
    flock 9
    mkdir -p "${PYTHONUSERBASE}" "${PIP_CACHE_DIR}"
    if [ ! -f "${PYTHONUSERBASE}/.deps-installed" ] || [ pyproject.toml -nt "${PYTHONUSERBASE}/.deps-installed" ]; then
      singularity exec --nv "${CUDAQ_IMAGE}" "${PYTHON}" -m pip install --user -e .
      touch "${PYTHONUSERBASE}/.deps-installed"
    fi
  ) 9>"${PYTHONUSERBASE}.lock"
fi

singularity exec --nv "${CUDAQ_IMAGE}" "${PYTHON}" --version
singularity exec --nv "${CUDAQ_IMAGE}" "${PYTHON}" -c 'import cudaq; print(cudaq.__version__)'

singularity exec --nv "${CUDAQ_IMAGE}" "${PYTHON}" "${SCRIPT}" ${SCRIPT_ARGS:-}
