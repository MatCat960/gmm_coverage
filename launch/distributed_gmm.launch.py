#!/usr/bin/env python3
#


import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import ThisLaunchFileDir
from launch_ros.actions import Node
from ament_index_python.packages import get_package_prefix



def generate_launch_description():
    TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']
    ROBOTS_NUM = 5
    ROBOT_RANGE = 10.0
    AREA_SIZE_x = 20.0
    AREA_SIZE_y = 20.0
    AREA_LEFT = -10.0
    AREA_BOTTOM = -10.0
    SIM = True
    GUI = False

    ld = []
    for i in range(0,ROBOTS_NUM):
        n = Node(
            package='gmm_coverage',
            node_executable='distributed_gmm',
            name='distributed_gmm_'+str(i),
            remappings=[('/gaussian_mixture_model', '/gaussian_mixture_model_'+str(i))],
            parameters=[{"ROBOTS_NUM": ROBOTS_NUM}, {"ROBOT_RANGE": ROBOT_RANGE}, {"AREA_SIZE_x": AREA_SIZE_x}, {"AREA_SIZE_y": AREA_SIZE_y}, {"AREA_LEFT": AREA_LEFT}, {"AREA_BOTTOM": AREA_BOTTOM}, {"SIM": SIM}, {"ID": i}, {"GUI": GUI}],
            output='screen')

        ld.append(n)

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
        ld.append(rviz)

        gmm_visualizer = Node(
            package='gmm_coverage',
            node_executable='gmm_visualizer',
            name='gmm_visualizer',
            remappings=[('/gaussian_mixture_model', '/gaussian_mixture_model_0')],
            output='screen'
        )
        ld.append(gmm_visualizer)

    return LaunchDescription(ld)