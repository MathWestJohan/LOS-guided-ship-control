# LOS-guided-ship-control

Project repository for course MMA4007 Applied AI and Control.

A ship autopilot simulation using Line-Of-Sight (LOS) guidance with PID control in the [AGX Dynamics](https://www.algoryx.se/agx-dynamics/) physics engine. The ship follows a set of waypoints using classic marine guidance and control theory. The goal is to collect training data for an ML-assisted autonomous docking system. The guidance points, physical dock point and maneuvering methods are yet to be implemented. Currently, this is simply a LOS PID guidance system following 3 waypoints before finishing the run.

## How it works

LOS Guidance computes the desired heading based on the ship's cross-track error relative to the path between waypoints. Reference filters smooth the heading and speed commands. A PID controller produces surge, sway, and yaw forces/moments. Thrust allocation distributes the commands to two stern azimuth thrusters. A state observer filters noisy position measurements and estimates velocities. 

The simulation logs all states, references, and control signals to a CSV file for later analysis and ML training.

## Requirements

- AGX Dynamics with Python bindings
- Python 3.9.9
- Numpy, Matplotlib
