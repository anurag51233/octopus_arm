'''
Author: David Valencia (Modified for 21-joint configuration)
Date: 2026

Describer:  An action Client to move the robot joint to a specific position
            Updated to match Plane_002 through Plane_022 joints.
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
        # Topic name updated to match standard controller convention
        self.action_client = ActionClient(self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')

    def send_goal(self):
        self.get_logger().info("Sending Goal to the Action Server")
        
        # 1. Define the joint names exactly as they appear in your YAML
        joint_names = [
            'Plane_002_joint', 'Plane_003_joint', 'Plane_004_joint', 'Plane_005_joint',
            'Plane_006_joint', 'Plane_007_joint', 'Plane_008_joint', 'Plane_009_joint',
            'Plane_010_joint', 'Plane_011_joint', 'Plane_012_joint', 'Plane_013_joint',
            'Plane_014_joint', 'Plane_015_joint', 'Plane_016_joint', 'Plane_017_joint',
            'Plane_018_joint', 'Plane_019_joint', 'Plane_020_joint', 'Plane_021_joint',
            'Plane_022_joint'
        ]

        # 2. Create the trajectory points
        # Since you have 21 joints, each positions list must have exactly 21 values
        points = []

        # Point 1: Home/Zero Position
        point1_msg = JointTrajectoryPoint()
        point1_msg.positions = [0.0] * 21 
        point1_msg.time_from_start = Duration(seconds=2.0).to_msg()
        points.append(point1_msg)

        # Point 2: Example movement (moving the first few joints)
        point2_msg = JointTrajectoryPoint()
        # Initializing 21 zeros and modifying specific indices
        pos2 = [0.0] * 21
        pos2[0] = 0.5  # Plane_002_joint
        pos2[2] = 0.52 # Plane_004_joint
        pos2[4] = 0.17 # Plane_006_joint
        pos2[6] = -0.17 # Plane_008_joint
        pos2[8] = -0.52 # Plane_010_joint
        pos2[10] = 0.17 # Plane_012_joint
         # You can set other joints as needed, just ensure all 21 are defined
         
        point2_msg.positions = pos2
        point2_msg.time_from_start = Duration(seconds=5, nanoseconds=0).to_msg()
        points.append(point2_msg)

        # 3. Create and Send the Goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.goal_time_tolerance = Duration(seconds=1, nanoseconds=0).to_msg()
        goal_msg.trajectory.joint_names = joint_names
        goal_msg.trajectory.points = points

        self.get_logger().info("Waiting for action server...")
        self.action_client.wait_for_server()
        
        self.send_goal_future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: ' + str(result))
        # Optional: shutdown here or stay alive to send more goals
        # rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        # You can access feedback.actual, feedback.desired, etc.
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