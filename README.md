# Deep Reinforcement Learning with Dynamic Graphs for Adaptive Informative Path Planning

Public version of Dynamic Graph based adaptive IPP approach code

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

<img src=dynaGr.gif width=711 height=400>

### Python Environment

One test instance in python environment

<img src=https://github.com/AccGen99/3d_ipp/blob/main/%20uav_1.gif width=711 height=400>
