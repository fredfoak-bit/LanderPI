from glob import glob
from setuptools import setup

package_name = 'HRI_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    description='Hand gesture recognition and response',
    entry_points={
        'console_scripts': [
            'fist_back_node = HRI_pkg.HRI:main', #for hand gesture recognition
        ],
    },
)
