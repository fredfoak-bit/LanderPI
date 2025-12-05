import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Start the drivers (Camera, Chassis, Lidar)
    # We can reuse your existing launch file to start all hardware drivers
    # This prevents code duplication.
    #
    drivers_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('green_nav_pkg'), 'launch'),
            '/green_nav_with_sensors.launch.py'
        ])
    )

    # 2. Start the Fist Detection Node
    fist_node = Node(
        package='green_nav_pkg',
        executable='fist_back_node',
        name='fist_back_node',
        output='screen',
    )

    return LaunchDescription([
        drivers_launch,
        fist_node
    ])