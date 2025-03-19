#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, EnvironmentVariable, PathJoinSubstitution
from launch_ros.actions import Node



def generate_launch_description():
    uav_name = EnvironmentVariable('UAV_NAME')

    ns = LaunchConfiguration('uav_name',default=uav_name)
    config_dir = os.path.join(get_package_share_directory('gmm_coverage'), 'config')
    config_arg = DeclareLaunchArgument(
        'config',
        description='Name of the parameter file (with extension)'
    )
    param_file = PathJoinSubstitution([
        config_dir,
        LaunchConfiguration('config')
    ])
    node = Node(
        package='gmm_coverage',
        executable='distributed_gmm',
        namespace=ns,
        name='team_gmm_coverage',
        parameters=[param_file],
        output='screen'
    )


    return LaunchDescription([config_arg,node])
