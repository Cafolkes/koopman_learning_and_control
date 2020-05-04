# koopman-learning-control
Python simulation and hardware library for Koopman based learning and control

The simulation framework of this repository is utilizing the [Learning and Control Core Library](https://github.com/learning-and-control/core).

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
