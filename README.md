# ORNL Workshop


## Running CUDA Quantum on Ascent
Ascent has NVIDIA V100 GPUs and you should be able to login and submit jobs that leverage them. If you have access to the Ascent system, CUDA Quantum has been built from source and is available as a module. 

1. Clone the workshop repo  \
`$ git clone https://github.com/justinlietz/ornl-cudaq-workshop` 
 
2. Upload the module \
```$source batch_scripts/load_env.sh ```

3. Running examples in demos/examples in C++ or Pythong.


## Running CUDA Quantum for non-OLCF systems
If you do not have access to Ascent, the easiest way to get started with CUDA Quantum is via the public Docker images. These images are available for `x86_64` (or `AMD64`) and `aarch64` CPU architectures. Depending on whether you have an NVIDIA GPU or not, you’ll also want to pull a specific tag for these images. 
 
x86_64 / AMD64, `ghcr.io/nvidia/cuda-quantum:latest` 
 
Please pull these images before you arrive at the workshop (using, `docker pull $IMAGE`). If your laptop is based on an ARM processor (e.g. like the newest Macbooks, M1 / M2), make sure you pull the AARCH64 image.

To pull these images, you will need to install [docker](https://www.docker.com/) and then run `docker pull <image_name>`.\
For instructions on how to run the CUDA Quantum container, refer to [https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda-quantum](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/quantum/containers/cuda-quantum).
 
CUDA Quantum programs run natively via backend-extensible circuit simulators. The most performant of these require an NVIDIA GPU (e.g. V100, A100, H100, A6000, A4000, etc.). If you do not have access to such a GPU (e.g. on your Macbook), then you will not be able to target these backends. If you have access to a remote workstation with an NVIDIA GPU that you can access during the tutorial, that would be best.

## Targets
A `--target <target-name>` flag can be specified at compilation for `C++` and at runtime for `Python`, which is a combination of the desired platform and simulator / QPU. 
To get additional information on the simulators and backends, go to [demos/TARGETS.md](demos/TARGETS.md).
