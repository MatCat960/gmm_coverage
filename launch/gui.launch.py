import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_path = get_package_share_directory('gmm_coverage')
    config_path = os.path.join(pkg_path, 'rviz/gmm.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', config_path],
        output='screen'
    )


    gmm_visualizer = Node(
        package='gmm_coverage',
        executable='gmm_visualizer',
        name='gmm_visualizer',
        remappings=[('/gaussian_mixture_model', '/gaussian_mixture_model_0')],
        output='screen'
    )
    LaunchDescription([ rviz, gmm_visualizer])
