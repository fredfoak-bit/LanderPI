import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Start the drivers
    drivers_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('HRI_pkg'), 'launch'),
            '/sensors.launch.py'
        ])
    )

    # 2. Start the TTS Node (REQUIRED for speech)
    # This assumes you have the 'large_models' package in your workspace
    tts_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('large_models'), 'launch'),
            '/tts_node.launch.py'
        ])
    )

    # 3. Start the Fist Detection Node
    fist_node = Node(
        package='HRI_pkg',
        executable='fist_back_node',
        name='fist_back_node',
        output='screen',
    )

    return LaunchDescription([
        drivers_launch,
        tts_launch,
        fist_node
    ])