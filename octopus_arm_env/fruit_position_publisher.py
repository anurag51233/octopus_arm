"""
Fruit World Position Publisher
================================
Author: David Valencia
Date:   2026

Subscribes to /gazebo/model_states (published by Gazebo automatically) and
re-publishes the world-frame pose of the 'red_fruit' model as a
geometry_msgs/PointStamped on /fruit_world_position.

This is the ground-truth signal used by the RL reward function to check
whether the octopus arm has successfully moved the fruit to the desired
drop location.

Topics published:
    /fruit_world_position  (geometry_msgs/PointStamped)  — fruit XYZ in world frame
    /fruit_target_position (geometry_msgs/PointStamped)  — fixed target XYZ

Run standalone:
    ros2 run octopus_arm_env fruit_position_publisher

Or import the node class and add it to an existing executor.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PointStamped

import numpy as np


# ---------------------------------------------------------------------------
# Configuration — adjust to match your world SDF
# ---------------------------------------------------------------------------

# The exact model name used in the SDF / world file
FRUIT_MODEL_NAME = 'red_fruit'

# Where the fruit starts in the world (matches your SDF <pose>)
FRUIT_INITIAL_POSITION = np.array([3.0, 0.0, 2.5], dtype=np.float64)

# Where the arm should deliver the fruit  ← CHANGE THIS to your desired drop point
FRUIT_TARGET_POSITION  = np.array([-3.0, 0.0, 1.0], dtype=np.float64)

# Publish rate (Hz)
PUBLISH_RATE_HZ = 20.0


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class FruitPositionPublisher(Node):
    """
    Reads Gazebo model states and publishes:
        /fruit_world_position  — current XYZ of the fruit in world frame
        /fruit_target_position — fixed target XYZ the fruit should reach
    """

    def __init__(self):
        super().__init__('fruit_position_publisher')

        # Best-effort QoS matches Gazebo's /model_states publisher
        gazebo_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscriber ─────────────────────────────────────────────────────
        self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self._model_states_cb,
            gazebo_qos,
        )

        # ── Publishers ─────────────────────────────────────────────────────
        self._fruit_pub  = self.create_publisher(PointStamped, '/fruit_world_position',  10)
        self._target_pub = self.create_publisher(PointStamped, '/fruit_target_position', 10)

        # ── Internal state ─────────────────────────────────────────────────
        self._fruit_position = FRUIT_INITIAL_POSITION.copy()
        self._fruit_found    = False

        # ── Periodic target publisher (so the env can always read target) ──
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._publish_target)

        self.get_logger().info(
            f"FruitPositionPublisher ready.  "
            f"Watching model '{FRUIT_MODEL_NAME}'.  "
            f"Target: {FRUIT_TARGET_POSITION.tolist()}"
        )

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _model_states_cb(self, msg: ModelStates):
        """Extract the fruit pose from /gazebo/model_states."""
        if FRUIT_MODEL_NAME not in msg.name:
            if not self._fruit_found:
                self.get_logger().warn(
                    f"Model '{FRUIT_MODEL_NAME}' not found in /gazebo/model_states. "
                    "Check your SDF model name."
                )
            return

        self._fruit_found = True
        idx = msg.name.index(FRUIT_MODEL_NAME)
        pose = msg.pose[idx]

        self._fruit_position = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ], dtype=np.float64)

        # Publish immediately on every model_states update for low latency
        self._publish_fruit()

    def _publish_fruit(self):
        msg = PointStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.point.x = float(self._fruit_position[0])
        msg.point.y = float(self._fruit_position[1])
        msg.point.z = float(self._fruit_position[2])
        self._fruit_pub.publish(msg)

    def _publish_target(self):
        msg = PointStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.point.x = float(FRUIT_TARGET_POSITION[0])
        msg.point.y = float(FRUIT_TARGET_POSITION[1])
        msg.point.z = float(FRUIT_TARGET_POSITION[2])
        self._target_pub.publish(msg)

    # ── Public getter (for use when node is embedded in env executor) ──────

    def get_fruit_position(self) -> np.ndarray:
        """Returns the latest known fruit world position as (3,) float64."""
        return self._fruit_position.copy()

    @staticmethod
    def get_target_position() -> np.ndarray:
        """Returns the fixed fruit drop-target position as (3,) float64."""
        return FRUIT_TARGET_POSITION.copy()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = FruitPositionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()