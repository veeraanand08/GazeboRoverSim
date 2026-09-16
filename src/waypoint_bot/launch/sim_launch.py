import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('waypoint_bot')
    world_path = os.path.join(pkg_share, 'worlds', 'rover_world.sdf')
    bridge_config = os.path.join(pkg_share, 'config', 'bridge.yaml')

    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py',
            )
        ),
        launch_arguments={'gz_args': f'-r {world_path}'}.items(),
    )

    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        arguments=['--ros-args', '-p', f'config_file:={bridge_config}'],
    )

    nav_node = Node(
        package='waypoint_bot',
        executable='astar_nav_node',
        name='astar_nav_node',
        output='screen',
        parameters=[{'num_waypoints': 3}],
    )

    return LaunchDescription([
        gz_sim_launch,
        bridge_node,
        nav_node,
    ])
