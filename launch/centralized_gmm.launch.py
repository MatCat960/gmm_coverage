#!/usr/bin/env python3
#
# 


import os
import random 

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_prefix

TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']


def generate_launch_description():
    TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']
    ROBOTS_NUM = 4
    ROBOT_RANGE = 4.0
    AREA_SIZE_x = 3.0
    AREA_SIZE_y = 3.0
    AREA_LEFT = -1.0
    AREA_BOTTOM = -1.0
    GUI = False
    SIM = False

    launch_list = []

    n = Node(
        package='gmm_coverage',
        node_executable='centralized_gmm',
        name='centralized_gmm',
        parameters=[{"ROBOTS_NUM": ROBOTS_NUM},
                    {"ROBOT_RANGE": ROBOT_RANGE},
                    {"GRAPHICS_ON": GUI},
                    {"SIM": SIM},
                    {"AREA_SIZE_x": AREA_SIZE_x},
                    {"AREA_SIZE_y": AREA_SIZE_y},
                    {"AREA_LEFT": AREA_LEFT},
                    {"AREA_BOTTOM": AREA_BOTTOM}],
        output='screen')

    launch_list.append(n)

    if GUI:
        pkg_path = get_package_prefix('gmm_coverage')
        config_path = os.path.join(pkg_path, '..', '..', 'src', 'gmm_coverage')
        config_path = os.path.join(config_path, 'rviz/gmm.rviz')

        rviz = Node(
            package='rviz2',
            node_executable='rviz2',
            name='rviz2',
            arguments=['-d', config_path],
            output='screen'
        )
        launch_list.append(rviz)

        gmm_visualizer = Node(
            package='gmm_coverage',
            node_executable='gmm_visualizer',
            name='gmm_visualizer',
            output='screen'
        )
        launch_list.append(gmm_visualizer)

    return LaunchDescription(
        launch_list
    )