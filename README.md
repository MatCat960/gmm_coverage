
# gmm_coverage #

  

This repository contains C++ and Python code to define and handle Gaussian Mixture Models for robots formation control.

  

## What is this repository for? ###

  

* Definition of Gaussian Mixture Models from a polygon drawn on a graphical interface.
* Control software to drive robots towards the region of interest.

  

## Requirements ##
 
### General ###
* Install custom ROS msgs for the definition of Gaussian Mixture Models:

    `git clone git@bitbucket.org:mcatellani96/gmm_msgs.git`


### C++ implementation ###

* Install ***SFML*** libray (for further information: https://www.sfml-dev.org/tutorials/2.5/start-linux.php):

	`sudo apt-get install libsfml-dev`

### Python implementation ###
* Install ***roipoly*** library for the definition of a region of interest (see https://github.com/jdoepfert/roipoly.py):   
	`pip install roipoly`
* Install ***scikit-learn*** library for  (for further information: https://scikit-learn.org/stable/index.html):   
	`pip install -U scikit-learn`
	

  

## How to use ##

* Launch an empy Gazebo world with desired number of robots spawning in random positions:

	`ros2 launch gmm_coverage world_gmm.launch.py`

 * Define the region of interest and calculate the Gaussian Mixture Model fitting the desired shape. Left click with the mouse to define a vertex of the polygon, right click to define the last vertex, stop drawing and calculate the Gaussian Mixture Model.
	 * **Python implementation**   
	 `ros2 run gmm_coverage interface.py`
	 
	 * **C++ implementation**   
	 `ros2 run gmm_coverage hs_interface`
	 
	 *Note:* when using the C++ implementation, a warning window may appear. Don't close the window, otherwise the interface will consider the mouse click as the definition of a new vertex.

* Check if the GMM is being published by the interface node on the `/gaussian_mixture_model` topic:   
	`ros2 topic list`

* Launch control nodes to drive robots:
	`ros2 launch gmm_coverage distributed_gmm.launch.py` (1 node for each robot)   

	or   

	`ros2 launch gmm_coverage centralized_gmm.launch.py`(1 node controlling every robot)   

	*Note:* Before launching nodes to control robots, make sure to check that parameters inside the launch file match the number of robots and environment size defined when spawning robots.
	 

A visualization tool is also provided to visualize Gaussian components of the Mixture. Set the `GUI` parameter in the launch file to `True` to launch it together with control nodes. A predefined RViz configuration will be loaded, showing each component as an ellipse colored according to its weight (green for small values, red for high values), and tracking robots during the mission. If you want to launch it separately:   
`ros2 run gmm_coverage gmm_visualizer`
