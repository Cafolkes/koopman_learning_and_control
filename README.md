# koopman-learning-and-control
Python simulation and hardware library for Koopman based learning and control.

The simulation framework of this repository is utilizing the [Learning and Control Core Library](https://github.com/learning-and-control/core).

## Code examples for bilinear EDMD learning and feedback linearizing control

This repository contains code to generate the examples in the paper

1. C. Folkestad, J. Burdick, "Koopman based Learning and Control of Control-affine Nonlinear Robotic Systems", in *Proc. Conference on Robotic Learning*, (submitted) 2020 

Specifically, the code used for numerical experiments are contained in three Jupyter notebooks:
1. bilinearizable_sys_fl_ipynb: Code containing example of a system that has a Koopman bilinear transform that can be computed analytically, and how to design feedback linearizing control law for the exact Koopman bilinear form.
2. planar_quadrotor_simple: Code containing planar quadrotor example with data collection, model learning, and control design. This notebook contains a simplified learning scheme for the bEDMD method that is computationally faster but does not guarantee that the learned model is feedback linearizable. In the example given the unconstrained learning problem works well, and this notebook is a good starting point as the code is simpler and faster to work with.
3. planar_quadrotor_fl: Code containing planar quadrotor example with data collection, model learning, and control design. This notebook contains the full learning scheme for the bEDMD method that guarantees that the learned model is feedback linearizable. In the example the bEDMD algorithm takes about 10 minutes to complete, depending on computational hardware.

## macOS setup
Set up virtual environment
```
python3 -m venv .venv
```
Activate virtual environment
```
source .venv/bin/activate
```
Upgrade package installer for Python
```
pip install --upgrade pip
```
Install requirements
```
pip3 install -r requirements.txt
```
Create IPython kernel
```
python3 -m ipykernel install --user --name .venv --display-name "Virtual Environment"
```
