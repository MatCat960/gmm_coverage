#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch_ros.actions import Node



def generate_launch_description():
    uav_name = EnvironmentVariable('UAV_NAME')

    ns = LaunchConfiguration('uav_name',default=uav_name)
    pkg_path = get_package_share_directory('gmm_coverage')
    param = os.path.join(pkg_path, 'config','params.yaml')
    node = Node(
        package='gmm_coverage',
        executable='individual_gmm_coverage',
        namespace=ns,
        name='individual_gmm_coverage',
        parameters=[param],
        output='screen'
    )


    return LaunchDescription([node])
