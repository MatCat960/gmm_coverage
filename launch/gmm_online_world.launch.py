#!/usr/bin/env python3

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

TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']

AREA_W = 10.0
ROBOTS_NUM = 8
TARGETS_NUM = 4


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    launch_file_dir = os.path.join(get_package_share_directory('gazebo_ros'), 'launch')
    launch_list = []

    gazebo = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_file_dir, '/gazebo.launch.py']),
            launch_arguments={'gui': 'true'}.items(),
        )

    launch_list.append(gazebo)


    x_pos = []
    y_pos = []
    theta = []

    for i in range(ROBOTS_NUM):
        ranx = random.uniform(-0.5*AREA_W, 0.5*AREA_W)
        rany = random.uniform(-0.5*AREA_W, 0.5*AREA_W)
        th = random.uniform(0,6)

        x_pos.append(ranx)
        y_pos.append(rany)
        theta.append(th)
        # print(th)

        

    for i in range(ROBOTS_NUM):
        spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                            arguments=['-entity', 'turtlebot'+str(i), '-database', 'turtlebot3_' + TURTLEBOT3_MODEL, '-robot_namespace', 'turtlebot'+str(i),
                                    '-x', str(float(x_pos[i])), '-y', str(float(y_pos[i])), '-z', str(0.1), '-Y', str(theta[i]),
                            ],
                            output='screen')

        launch_list.append(spawn_entity)


    xtargets = []
    ytargets = []
    for i in range(TARGETS_NUM):
        xt = random.uniform(-0.5*AREA_W, 0.5*AREA_W)
        yt = random.uniform(-0.5*AREA_W, 0.5*AREA_W)
        xtargets.append(xt)
        ytargets.append(yt)
        target = Node(package='gazebo_ros', executable='spawn_entity.py',
                            arguments=['-entity', 'target'+str(i), '-database', 'target', '-robot_namespace', 'target'+str(i),
                                    '-x', str(float(xt)), '-y', str(float(yt)), '-z', str(0.2),
                            ],
                            output='screen')

        launch_list.append(target)

    for i in range(TARGETS_NUM):
        target_pub = Node(
            package='gmm_coverage',
            node_executable='target_publisher_node',
            name='target_pub',
            parameters=[{"XT": xtargets[i]},
                        {"YT": ytargets[i]},
                        {"ID": i}],
            output='screen')
    
        launch_list.append(target_pub)
        

    return LaunchDescription(launch_list)