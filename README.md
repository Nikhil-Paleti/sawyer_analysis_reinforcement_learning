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
Demonstartion of the learnt model: https://www.youtube.com/watch?v=UYUPgIz7v30  


# Steps to reproduce the environment:

1. Install Cuda 11.3 and the corresponding cudnn version 
2. Insiall mujoco 2.1.0 
3. Install Conda
4. Clone the base environment
5. Install mujoco-py
6. clone the robosuite repository and follow their installation steps
7. Install pytorch 

# Model for simulation

The model for simulation is taken from:
https://github.com/vikashplus/sawyer_sim


# Known issues:
1. Continuing training by loading models is not complete. We are not saving the buffer so loading only networks and training is not providing good results.


# citations 
@inproceedings{todorov2012mujoco,
  title={Mujoco: A physics engine for model-based control},
  author={Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle={2012 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages={5026--5033},
  year={2012},
  organization={IEEE}
}

@inproceedings{robosuite2020,
  title={robosuite: A Modular Simulation Framework and Benchmark for Robot Learning},
  author={Yuke Zhu and Josiah Wong and Ajay Mandlekar and Roberto Mart\'{i}n-Mart\'{i}n},
  booktitle={arXiv preprint arXiv:2009.12293},
  year={2020}
  
}


