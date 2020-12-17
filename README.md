# simulators

A common repository for simulators for processes modeled by graph-based Markov models and related utilities to support 
various research projects. 

Current projects using this repository:
- [ddrl-firefighting](https://github.com/rhaksar/ddrl-firefighting): Deep reinforcement learning approach for 
  coordinating a team of autonomous aerial vehicles to fight a forest wildfire.
- [control-large-GMDPs](https://github.com/rhaksar/control-large-GMDPs): Approximate dynamic programming approach for 
  computing constrained policies to fight a forest wildfire and to limit the outbreak of a virus.
- [control-percolation](https://github.com/rhaksar/control-percolation): Percolation-based approach for computing 
  constrained policies to fight a forest wildfire. 

## Installation

Clone this repository and run `pip3 install -e .` in the directory in order to install the package and access the
simulators system-wide.

## Directories:
- `simulators/epidemics`: Simulate a disease epidemic.
- `simulators/fires`: Simulate a forest fire.

## Files:
- `simulators/Element.py`: Template for simulation elements. 
- `simulators/Simulator.py`: Template for simulators. 
- `examples/epidemicsExample.py`: Example use of the 2014 West Africa Ebola outbreak simulator.
- `examples/firesExample.py`: Example use of the lattice-based forest. 
