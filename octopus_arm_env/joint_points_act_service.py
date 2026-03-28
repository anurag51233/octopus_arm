'''
Author: David Valencia (Modified for 10-joint configuration)
Date: 2026

Describer:  An action Client to move the robot joint to a specific position
            Updated to match Plane_002 through Plane_011 joints.
'''

import rclpy
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

class TrajectoryActionClient(Node):

    def __init__(self):
        super().__init__('points_publisher_node_action_client')
        # Topic name must match standard controller convention
        self.action_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/joint_trajectory_controller/follow_joint_trajectory'
        )

    def send_goal(self):
        self.get_logger().info("Sending Goal to the Action Server")
        
        # 1. Define the joint names - MUST match your YAML exactly
        joint_names = [
            'Plane_002_joint', 'Plane_003_joint', 'Plane_004_joint', 'Plane_005_joint',
            'Plane_006_joint', 'Plane_007_joint', 'Plane_008_joint', 'Plane_009_joint',
            'Plane_010_joint', 'Plane_011_joint'
        ]

        num_joints = len(joint_names)
        points = []

        # Point 1: Home/Zero Position
        point1_msg = JointTrajectoryPoint()
        point1_msg.positions = [0.0] * num_joints 
        point1_msg.time_from_start = Duration(seconds=2, nanoseconds=0).to_msg()
        points.append(point1_msg)

        # Point 2: Example movement
        point2_msg = JointTrajectoryPoint()
        # Initializing list to match the length of joint_names
        pos2 = [0.0] * num_joints
        
        # Indexing starts at 0 (Plane_002_joint is index 0)
        pos2[0] = 1.57   # Plane_002_joint
        pos2[1] = -0.57 # Plane_003_joint   
        pos2[2] = 1.52  # Plane_004_joint
        pos2[3] = -0.52 # Plane_005_joint
        pos2[4] = 0.17  # Plane_006_joint
        pos2[5] = 1.00  # Plane_007_joint
        pos2[6] = -0.17 # Plane_008_joint
        pos2[7] = 0.17  # Plane_009_joint
        pos2[8] = 0.52  # Plane_010_joint
        pos2[9] = -0.52 # Plane_011_joint
        

        point2_msg.positions = pos2
        point2_msg.time_from_start = Duration(seconds=5, nanoseconds=0).to_msg()
        points.append(point2_msg)

        # 3. Create and Send the Goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = joint_names
        goal_msg.trajectory.points = points

        self.get_logger().info("Waiting for action server...")
        self.action_client.wait_for_server()
        
        self._send_goal_future = self.action_client.send_goal_async(
            goal_msg, 
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by the Action Server!')
            return

        self.get_logger().info('Goal accepted!')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Action finished with result code: {result.error_code}')

    def feedback_callback(self, feedback_msg):
        # Optional: Log the current position of the first joint
        # self.get_logger().info(f"Current Pos: {feedback_msg.feedback.actual.positions[0]}")
        pass

def main(args=None):
    rclpy.init(args=args)
    action_client = TrajectoryActionClient()
    action_client.send_goal()
    try:
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        pass
    finally:
        action_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()