**CUDA-Q Hands-on Workshop Materials**
=====================================

Welcome to the CUDA-Q Hands-on Workshop Materials repository! This repository is dedicated to collecting and organizing CUDA-Q code tutorials and materials from the many workshops given to various institutions across the world. The goal is to provide a comprehensive resource for learners and professionals alike, covering a range of beginner to advanced use cases.

Happy learning with CUDA-Q!

### Branch

Materials for each materials are at the branch.
For exapmle, cloning can be done by specifying a branch as follows:
```sh
git clone -b 202410-nagoya --single-branch https://github.com/cudaq-libraries/workshops.git
```

### Environment setup

```sh
pjsub --interact -L rscgrp=cx-interactive,jobenv=singularity,elapse=3:00:00

module load singularity
singularity exec --nv docker://nvcr.io/nvidia/quantum/cuda-quantum:0.8.0 jupyter lab --notebook-dir=${HOME} --ip='*' --port=8888 --no-browser --allow-root
```

Second terminal:
```sh
ssh -L 8888:cx064:8888 <USERNAME>@flow-cx.cc.nagoya-u.ac.jp
```

The cx064, username, and port number need to be appropriately replaced.
