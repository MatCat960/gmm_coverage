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
    ID = 0
    ROBOTS_NUM = 5
    NUM_SAMPLES = 300
    TARGETS_NUM = 4
    SENS_RANGE = 5.0
    COMM_RANGE = 7.0
    NOISE_DEV = 0.5
    ENV_SIZE = 20.0
    GRAPHICS_ON = True

    GUI = False

    launch_list = []

    for i in range(ROBOTS_NUM):
        n = Node(
            package='gmm_coverage',
            node_executable='gmm_calc',
            name='gmm_calculator',
            parameters=[{"ID": i},
                        {"NUM_SAMPLES": NUM_SAMPLES},
                        {"TARGETS_NUM": TARGETS_NUM},
                        {"SENS_RANGE": SENS_RANGE},
                        {"COMM_RANGE": COMM_RANGE},
                        {"NOISE_DEV": NOISE_DEV},
                        {"ENV_SIZE": ENV_SIZE},
                        # {"GUI", GUI},
                        {"GRAPHICS_ON": GRAPHICS_ON}],
            output='screen')
        
        GRAPHICS_ON = False

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
            remappings=[('/gaussian_mixture_model', '/gaussian_mixture_model_0')],
            output='screen'
        )
        launch_list.append(gmm_visualizer)

    return LaunchDescription(
        launch_list
    )