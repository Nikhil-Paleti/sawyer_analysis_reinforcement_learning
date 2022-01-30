# Introduction

`Sawyer_analysis.ipynb` contains:
* Forward Kinematics
* Inverse Kinematics
* Manipulator Jacobian calculation
* Null Space motion
* yoshikawa's manipulability measure
* Trajectory planning 

`Sawyer_RL.ipynb` contains:

* Function to train a ddpg model to lift a cube using sawyer robot
* Function to visualize the trained model


# Steps to reproduce the environment:

1. Install Cuda 11.3 and the corresponding cudnn version 
2. Insiall mujoco 2.1.0 
3. Install Conda
4. Clone the base environment
5. Install mujoco-py
6. clone the robosuite repository and follow their installation steps
7. Install pytorch 