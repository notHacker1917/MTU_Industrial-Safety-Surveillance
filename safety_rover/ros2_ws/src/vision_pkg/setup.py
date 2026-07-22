from setuptools import setup, find_packages

package_name = 'vision_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Safety Rover Team',
    maintainer_email='team@safety-rover.local',
    description='DepthAI OAK-D vision pipeline with PPE compliance',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_node = vision_pkg.ros2_vision_node:main',
        ],
    },
)
