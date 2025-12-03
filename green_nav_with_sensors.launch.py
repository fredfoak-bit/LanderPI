import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    controller_pkg = get_package_share_directory('controller')
    peripherals_pkg = get_package_share_directory('peripherals')

    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(controller_pkg, 'launch', 'controller.launch.py'))
    )
    depth_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(peripherals_pkg, 'launch', 'depth_camera.launch.py'))
    )
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(peripherals_pkg, 'launch', 'lidar.launch.py'))
    )

    green_nav_node = Node(
        package='green_nav_pkg',         
        executable='green_nav',     
        name='green_nav',
        output='screen',
        parameters=[{'debug': False}],
        remappings=[
            ('~/image_result', 'green_nav/image_result'),
        ],
    )

    return LaunchDescription([
        controller_launch,
        depth_camera_launch,
        lidar_launch,
        green_nav_node,
    ])
