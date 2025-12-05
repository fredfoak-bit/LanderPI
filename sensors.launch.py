import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Get package paths
    controller_package_path = get_package_share_directory('controller')
    peripherals_package_path = get_package_share_directory('peripherals')
    servo_package_path = get_package_share_directory('servo_controller') # <--- ADD THIS

    # 1. Start Chassis Control (Motors, IMU, Odom)
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(controller_package_path, 'launch/controller.launch.py')),
    )
    
    # 2. Start Depth Camera (RGB + Depth)
    depth_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, 'launch/depth_camera.launch.py')),
    )

    # 3. Start Lidar
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, 'launch/lidar.launch.py')),
    )

    # 4. Start Servo Controller (REQUIRED FOR CAMERA MOVEMENT) <--- ADD THIS BLOCK
    servo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(servo_package_path, 'launch/servo_controller.launch.py')),
    )

    return LaunchDescription([
        controller_launch,
        depth_camera_launch,
        lidar_launch,
        servo_launch, # <--- Add to list
    ])