import os 
from glob import glob
from setuptools import setup

package_name = 'octopus_arm_env'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        # (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.rviz'))),    
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.*'))),
        (os.path.join('share', package_name, 'worlds'), glob(os.path.join('worlds', '*.world'))),           
        (os.path.join('share', package_name, 'description', 'urdf'), glob(os.path.join('description','urdf', '*.urdf'))),
        # (os.path.join('share', package_name, 'description', 'xacro'), glob(os.path.join('description','xacro', '*.xacro'))),
        (os.path.join('share', package_name, 'description', 'meshes'), glob(os.path.join('description','meshes', '*.obj'))),
        (os.path.join('share', package_name, 'description', 'meshes'), glob(os.path.join('description','meshes', '*.mtl'))),
        (os.path.join('share', package_name, 'description', 'meshes'), glob(os.path.join('description','meshes', '*.stl'))),
        # (os.path.join('share', package_name, 'description', 'meshes', 'm1013_white'), glob(os.path.join('description','meshes','m1013_white', '*.dae'))),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anu',
    maintainer_email='anu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'trajectory_points_topic = octopus_arm_env.joint_points_topic:main',
                'trajectory_points_act_server = octopus_arm_env.joint_points_act_service:main',
                'train_env = octopus_arm_env.env:main',
                'detector_node = octopus_arm_env.detector:main',
                'fruit_position_publisher = octopus_arm_env.fruit_position_publisher:main'
        ],
    },
)
