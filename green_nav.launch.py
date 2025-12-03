from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='green_nav_pkg',
            executable='green_nav',     
            parameters=[{'debug': False}],
            remappings=[
                ('~/image_result', 'green_nav/image_result'),
            ],
        )
    ])
