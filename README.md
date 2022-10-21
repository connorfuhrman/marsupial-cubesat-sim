# Decentralized Coordination of a Swarm of Nano-Spacecraft for Efficient Sample Return Using Visible Light Communication (VLC)

This repository holds the simulation code for this paper.


## Functionality
The `DSRC` package provides functionality to simulate an arbitrary number of motherships each with an arbitrary
number of cuesats which may be deployed, independently navigate, capture a sample, dock, and communicate.
Simulations are available as Ray `Actor`s which effectuate arbitrary task parallelism

**Note**: This is still under active development.

## QuickStart
*This probably won't work. Please add any packages you had to install manually to `requirements.txt` and issue a PR*

This repository is expected to be cross-platform as it is pure Python, however development is conducted mainly on MacOS and Ubuntu.

```python
git clone git@github.com:SpaceTREx-Lab/Decentralized-CubeSat-Sample-Return-using-VLC.git repodir
pip install -e repodir
python repodir/DSRC/simulation/simulation.py [--num_workers=<n parallel sims>]
open simulation_animation.mp4   # On MacOS
```

Passing the optional argument `--num_workers` with a value greater than 1 will launch the provided number of
Ray `Actor`s which will each independently run a simulation. 
The simulation test function sets the number of cubesats as increasing from 1 for each instantiated simulation.

The results of each simulation are used to create an animation.
This is saved to an MP4 by default but you can comment that portion out and replace with `plt.show()`
for an interactive animation (this is *far* faster to render at least on my laptop).



## Structure
The `DSRC` folder is the top-level for the package. 
Currently there is only the `simulation` subpacakge which holds the simulation class (and `simulation.py` which can be run as a test),
and the animation utilities.
The `simulation.spacecraft` subpackage holds classes for a spacecraft (a base class), a mothership, and a cubesat.
The `simulation.communication` holds utilities for simulating a communication link. (*This is still under development and untested*)
