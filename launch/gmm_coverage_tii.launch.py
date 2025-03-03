#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node



def generate_launch_description():
    ns = LaunchConfiguration('uav_name')
    pkg_path = get_package_share_directory('gmm_coverage')
    param = os.path.join(pkg_path, 'config','params.yaml')
    node = Node(
        package='gmm_coverage',
        executable='distributed_gmm',
        namespace=ns,
        name='distributed_gmm',
        parameters=[param],
        output='screen'
    )


    return LaunchDescription([node])
