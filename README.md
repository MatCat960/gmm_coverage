
# gmm_coverage #

  

This repository contains C++ and Python code to define and handle Gaussian Mixture Models for robots formation control.

  

## What is this repository for? ###

  

* Definition of Gaussian Mixture Models from a polygon drawn on a graphical interface.
* Control software to drive robots towards the region of interest.

  

## Requirements

* [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)
* Eigen3: `sudo apt install libeigen3-dev`
* [Tensorflow](https://www.tensorflow.org/install/pip)
* [Roads Detection Model](https://drive.google.com/file/d/1dfdPuzAOjxv7tyFnCo3qPSDg3BL5kKfM/view?usp=drive_link)
    (to be put inside `gmm_coverage/`)


## How to use it?

**Roads Detection from aerial image**

Use the trained neural network model to find roads in an aerial image.

`roslaunch gmm_coverage roads_gmm.launch`

Parameters:
- `filename` - name of the .png aerial image. The script searches for it inside `models` folder.
- `COMPONENTS_NUM ` - number of Gaussian components for the Gaussian Mixture Model.
- `SHOW_IMAGES` - show images resulting from detection (mask, GMM heatmap, GMM 3D model).
- `SAVE_IMAGES` - save resulting images.

**Spawn Gazebo world and UAV models**

Use [RotorS simulator](https://github.com/ethz-asl/rotors_simulator) to open a Gazebo world and spawn a desired number of UAVs in random starting positions.

`cd gmm_coverage/`

`./run.bash`

Main Parameters:
- `world_name` - name of the Gazebo world. Currently available: empty, desert, reggioemilia, colosseo.
- `gui` - whether to use the Gazebo graphical interface or not.
- `use_quad<n>` - UAVs selector. **Note:** quad20 must always be present because it becomes number 0.
- `x<n>`, `y<n>` - starting position of UAV `n`. This parameters are overwritten by the args in the bash script.

**Control UAVs and save logs**

Launch a controller node for each UAV. If desired, show a simplified graphical interface in RViz and save performance metrics in log file.

`roslaunch gmm_coverage gmm_distributed.launch`

Parameters:
- `ROBOTS_NUM` - number of UAVs to control (must be equal to the number of UAVs selected).
- `ROBOT_RANGE` - UAV's sensing range.
- `GUI` - whether to use RViz interface or not.
- `SIM` - use simulated or real robots.
- `AREA_SIZE_x` - width of the environment.
- `AREA_SIZE_y` - height of the environment.
- `AREA_LEFT` - x coordinate of global reference frame.
- `AREA_BOTTOM` - y coordinate of the global reference frame.
- `SAVE LOGS` - whether to evaluate performance metrics and save log files.
- `use_quad<n>` - UAVs selector. **Note:** quad20 must always be present because it becomes number 0.