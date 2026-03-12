"""
Author: David Valencia
Date: 25 / 08 /2021

Describer:  Simple launch to SIMULATE the doosan robot in GAZEBO in my own package
            Based on the original git package from doosan-robot2
            This scripts just spawns the robot arm in GAZEBO
            the robot description (urdf and xacro) are in: src/my_doosan_pkg/description/xacro

            Robot model m1013 color white.
            Robot model a0912 color blue.
"""

import os
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import Command
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # robot model to option m1013 or a0912

    robot_model = 'octopus_arm'
    # robot_model = 'm1013'


    urdf_file = os.path.join(
        get_package_share_directory('octopus_arm_env'),
        'description',
        'urdf',
        robot_model + '.urdf'
    )

    # Read URDF file
    with open(urdf_file, 'r') as file:
        robot_description = file.read()
        
    # Robot State Publisher
    robot_state_publisher = Node(package='robot_state_publisher',
                                 executable='robot_state_publisher',
                                 name='robot_state_publisher',
                                 output='both',
                                 parameters=[{'robot_description': robot_description}])

    # Spawn the robot in Gazebo
    spawn_entity_robot = Node(package='gazebo_ros',
                              executable='spawn_entity.py',
                              arguments=['-entity', 'octopus_arm', '-topic', 'robot_description'],
                              output='screen')
    #ros2 run gazebo_ros spawn_entity.py -topic robot_description -entity octopus_arm
    
    # Start Gazebo with my empty world
    world_file_name = 'my_empty_world.world'
    world = os.path.join(get_package_share_directory('octopus_arm_env'), 'worlds', world_file_name)
    gazebo_node = ExecuteProcess(cmd=['gazebo', '--verbose', world, '-s', 'libgazebo_ros_factory.so'], output='screen')

    # load and START the controllers in launch file

    load_joint_state_broadcaster = ExecuteProcess(
                                        cmd=['ros2', 'control', 'load_controller', '--set-state', 'start','joint_state_broadcaster'],
                                        output='screen')


    load_joint_trajectory_controller = ExecuteProcess( 
                                        cmd=['ros2', 'control', 'load_controller', '--set-state', 'start', 'joint_trajectory_controller'], 
                                        output='screen')

    #load_joint_state_broadcaster, load_joint_trajectory_controller

    return LaunchDescription([robot_state_publisher, spawn_entity_robot, gazebo_node, load_joint_state_broadcaster, load_joint_trajectory_controller])
