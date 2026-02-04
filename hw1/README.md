# 16-745 Assignment 1

This repository contains code for Assignment 1 of 16-745, covering function optimization and policy optimization for a two-wheeled inverted pendulum (TWIP).

## Setup

Create a Python environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

Install required packages:
```bash
pip install -r requirements.txt
```

## Files

- `part1b.py`  
  Optimizing a learned approximation of the 4D Rosenbrock function.

- `part2a.py`  
  Parametric policy optimization of a linear feedback controller for the two-wheeled inverted pendulum (TWIP).

- `part2b.py`  
  Neural network policy that learns feedback gains via behavior cloning.

- `sim_params.yaml`  
  Physical parameters and simulation settings for the TWIP environment.

- `models/`  
  Pretrained models for Parts 1b, 2a, and 2b (included in the submission).


## Notes

- Models are automatically loaded if present; otherwise they are trained and saved.
- Simulation parameters (masses, inertias, time step, initial state ranges) can be modified in `sim_params.yaml`.
- Each script generates plots at the end of execution for visualization and analysis.
- Random seeds are fixed where applicable to ensure reproducibility.
