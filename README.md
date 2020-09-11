# simulators

A common repository for simulators for processes modeled by graph-based Markov models and related utilities to 
support various research projects. 

## Installation

Clone this repository and run `pip3 install -e .` in the directory in order to install the package and access the
simulators system-wide.

## Directories:
- `epidemics`: Simulate a disease epidemic.
- `fires`: Simulate a forest fire.

## Files:
- `Element.py`: Template for simulation elements. 
- `Simulator.py`: Template for simulators. 
- `epidemicsExample.py`: Example use of the 2014 West Africa Ebola outbreak simulator.
- `firesExample.py`: Example use of the lattice-based forest. 
