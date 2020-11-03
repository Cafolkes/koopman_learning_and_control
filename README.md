# koopman-learning-and-control
Python simulation and hardware library for Koopman based learning and control.

The simulation framework of this repository is utilizing the [Learning and Control Core Library](https://github.com/learning-and-control/core).

## Code examples for bilinear EDMD learning and nonlinear model predictive control

This repository contains code to generate the examples combining learning of control-affine dynamics with Koopman bilinear models and nonlinear model predictive control. Specifically, the code used for numerical experiments are contained in two Jupyter notebooks:
1. bilinearizable_sys_mpc.ipynb: Code containing example of a system that has a Koopman bilinear transform that can be computed analytically, and how to design the nonlinear model predictive control law for the exact Koopman bilinear form.
2. planar_quadrotor_learning_mpc.ipynb: Code containing planar quadrotor example with data collection, model learning, and control design. 

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
