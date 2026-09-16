import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'waypoint_bot'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vanand1',
    maintainer_email='anandkaliyamurthy@gmail.com',
    description='Autonomous waypoint navigation with lidar-mapped A* planning for a simulated rover in Gazebo Harmonic',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'astar_nav_node = waypoint_bot.astar_nav_node:main',
        ],
    },
)
