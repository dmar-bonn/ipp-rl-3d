# Deep Reinforcement Learning with Dynamic Graphs for Adaptive Informative Path Planning

This repository contains the code of our paper "Deep Reinforcement Learning with Dynamic Graphs for Adaptive Informative Path Planning".
The paper can be found [here](https://ieeexplore.ieee.org/abstract/document/10578000).

If you found this work useful for your own research, feel free to cite it.
```commandline
@article{vashisth2024ral,
  author={Vashisth, Apoorva and R{\"u}ckin, Julius and Magistri, Federico and Stachniss, Cyrill and Popovic, Marija},
  journal={IEEE Robotics and Automation Letters (RA-L)}, 
  title={{Deep Reinforcement Learning with Dynamic Graphs for Adaptive Informative Path Planning}}, 
  year={2024},
  pages={1-8},
}
```
The codebase of this paper builds upon the work "CAtNIPP: Context-aware attention-based network for informative path planning" by Cao et al. published in the Proc. of the Conf. on Robot Learning (CoRL), 2023. 
We found their work and open-sourced code to be extremely helpful to conduct our research and advance RL-based IPP approaches. Please acknowledge this by citing their work as well:

```commandline
@inproceedings{cao2023catnipp,
  title={{CAtNIPP: Context-aware attention-based network for informative path planning}},
  author={Cao, Yuhong and Wang, Yizhuo and Vashisth, Apoorva and Fan, Haolin and Sartoretti, Guillaume Adrien},
  booktitle=corl,
  year={2023}
}
```

## Setting up code

* Define path for saving gifs and plots in env.py
* Python == 3.7
* Pytorch == 1.13
* Ray == 2.7

## Training

Train the model by running the following command -

```
python driver.py
```
To specify the number of parallel environment instances, change the variable ```NUM_META_AGENT``` in [```parameters.py```](parameters.py)

To specify the test environment change the variable ```TEST_TYPE``` in [```test_parameters.py```](test_parameters.py) to one of ```random``` or ```grid```.

You can change the range of target detection in sensor module by changing values of ```DEPTH``` parameter in [```parameters.py```](parameters.py). Note that the environment built in python considers an occupancy grid of 50 cells in each of 3 directions and the ```DEPTH``` variable specifies the depth of sensor frustum in terms of number of grid cells.

## Key Files

* driver.py - Driver of program. Holds global network.
* runner.py - Compute node for training. Maintains a single meta agent containing one instance of environment.
* worker.py - A single agent in a the IPP instance.
* parameter.py - Parameters for training and test.
* env.py - Define the environment class.

## Results

Evolution of Dynamic graph as episode progresses

<img src=gifs/dynaGr.gif width=711 height=400>

### Python Environment

One test instance in python environment

<img src=https://github.com/AccGen99/3d_ipp/blob/main/%20uav_1.gif width=711 height=400>

## Funding

This work was partially funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy - EXC
2070 – 390732324. A.V. is with the Department of Mechanical Engineering, Indian Institute of Technology, Kharagpur, India. J.R., F.M., C.S., 
and M.P. are with the Institute of Geodesy and Geoinformation, Cluster of Excellence PhenoRob, University of Bonn. M.P. is also with MAVLab, 
Faculty of Aerospace Engineering, TU Delft. C.S. is also with the University of Oxford and Lamarr Institute for Machine Learning and Artificial 
Intelligence, Germany.
