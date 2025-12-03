from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='landerpi',          # update to your package name
            executable='green_nav',      # console_script entry point -> green_nav:main
            name='green_nav',
            output='screen',
            parameters=[{'debug': False}],   # keep/override params as needed
            remappings=[
                ('~/image_result', 'green_nav/image_result'),
            ],
        )
    ])
