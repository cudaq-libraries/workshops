**CUDA-Q Hands-on Workshop Materials**
=====================================

Welcome to the CUDA-Q Hands-on Workshop Materials repository! This repository is dedicated to collecting and organizing CUDA-Q code tutorials and materials from the many workshops given to various institutions across the world. The goal is to provide a comprehensive resource for learners and professionals alike, covering a range of beginner to advanced use cases.

Happy learning with CUDA-Q!

### Branch

Materials for each materials are at the branch.
For exapmle, cloning can be done by specifying a branch as follows:
```sh
git clone -b main --single-branch https://github.com/cudaq-libraries/workshops.git
```

### Generating Scripts

The notebooks under `notebooks/` are the source of truth. The Python files under `scripts/` are generated from those notebooks with Jupytext, using the pairing rules in `pyproject.toml`.

- Edit notebooks first, then sync the paired scripts with Jupytext.
- Treat `scripts/` as generated output. Commit it only when a workshop branch explicitly needs those generated files.

To sync all notebooks:

```sh
find notebooks -name "*.ipynb" -print0 | xargs -0 jupytext --sync
```

To sync one notebook:

```sh
jupytext --sync notebooks/cudaq_introduction.ipynb
```
