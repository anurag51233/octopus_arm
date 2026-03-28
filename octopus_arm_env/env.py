"""
Octopus Arm Gazebo RL Environment  (v2 — Fruit Transport Task)
===============================================================
Author: David Valencia
Date:   2026

A Gymnasium-compatible environment wrapping the 10-joint octopus arm
simulated in Gazebo via ROS 2 Foxy.

Task:
    The octopus arm must grasp the red_fruit sphere (spawned at a fixed
    start position in the world) and transport it to a target drop location.

    The fruit's 3D position is detected in two complementary ways:
        1. Camera-based detection (ObjectDetector sub-node) — noisy but
           vision-realistic; included as part of the RL observation.
        2. Gazebo ground-truth (FruitPositionPublisher sub-node) — used
           ONLY for reward computation, not exposed to the policy.

Joints (10 total):
    Plane_002_joint ... Plane_011_joint

Action space:
    Box(10,)  — desired joint positions in radians, clipped to [JOINT_MIN, JOINT_MAX]

Observation space:
    Box(36,)
        [0  :10] — current joint positions          (rad)
        [10 :20] — current joint velocities         (rad/s)
        [20 :30] — target joint configuration       (rad)   ← arm pose goal
        [30 :33] — camera-detected fruit 3D pos     (m, in camera frame)
        [33 :36] — fruit drop-target position       (m, in world frame)

Reward  (fruit-transport shaped reward):
    r = w_arm   * -||q  - q_target||²          dense arm-pose penalty
      + w_fruit * -||p_fruit - p_target||²      dense fruit-to-goal penalty
      - time_penalty                             encourages speed
      + BONUS_FRUIT_AT_GOAL  when fruit reaches DROP_TOLERANCE of target
      + BONUS_ARM_AT_GOAL    when arm also reaches GOAL_TOLERANCE   (optional)

Episode termination:
    - Fruit within DROP_TOLERANCE of target AND arm within GOAL_TOLERANCE
    - Step count exceeds MAX_STEPS (timeout)

Dependencies:
    pip install gymnasium numpy opencv-python cv_bridge
    ROS 2 Foxy + gazebo_ros + control_msgs + sensor_msgs + gazebo_msgs

Nodes running inside this env (all share one MultiThreadedExecutor):
    _OctopusArmNode          — joint state subscriber + trajectory action client
    ObjectDetectorNode       — RGB-D camera → /detected_object_3d
    FruitPositionPublisher   — /gazebo/model_states → /fruit_world_position
"""

from __future__ import annotations

import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_srvs.srv import Empty
from sensor_msgs.msg import JointState, Image, CameraInfo
from geometry_msgs.msg import PointStamped
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from gazebo_msgs.msg import ModelStates
from rclpy.duration import Duration

from cv_bridge import CvBridge
import cv2

import gymnasium as gym
from gymnasium import spaces


# ============================================================================
# Global constants
# ============================================================================

# ── Arm ─────────────────────────────────────────────────────────────────────

JOINT_NAMES: list[str] = [
    'Plane_002_joint', 'Plane_003_joint', 'Plane_004_joint',
    'Plane_005_joint', 'Plane_006_joint', 'Plane_007_joint',
    'Plane_008_joint', 'Plane_009_joint', 'Plane_010_joint',
    'Plane_011_joint',
]
NUM_JOINTS      = len(JOINT_NAMES)
JOINT_MIN       = -np.pi / 10.2          # rad
JOINT_MAX       =  np.pi / 10.2          # rad
GOAL_TOLERANCE  = 0.05                  # rad — arm pose success criterion
MAX_STEPS       = 1                   # steps per episode
STEP_DURATION   = 0.2                    # wall-clock seconds between obs
TRAJECTORY_TIME = 5.0                   # seconds for controller
DELTA_MAX = 0.05                        # max radians per step

# ── Fruit transport ──────────────────────────────────────────────────────────

# Where the fruit is spawned in the world (must match your SDF <pose>)
FRUIT_START_POSITION  = np.array([3.0, 0.0, 2.5], dtype=np.float64)

# Where the arm must deliver the fruit  ← CHANGE TO YOUR DESIRED DROP LOCATION
FRUIT_TARGET_POSITION = np.array([-3.0, 0.0, 1.0], dtype=np.float64)

# Fruit must be within this distance (m) of the target to count as delivered
DROP_TOLERANCE = 0.3    # metres

# ── Reward weights ───────────────────────────────────────────────────────────

W_ARM_POSE      = 0.2   # weight on arm-to-joint-target distance penalty
W_FRUIT         = 1.0   # weight on fruit-to-drop-target distance penalty
TIME_PENALTY    = 0.5   # subtracted every step
BONUS_FRUIT     = 200.0 # fruit reaches drop zone
BONUS_ARM       = 50.0  # arm also at goal pose (optional secondary bonus)

# ── Misc ─────────────────────────────────────────────────────────────────────

RESET_SETTLE    = 0.5   # seconds after /reset_world before collecting obs
FRUIT_MODEL_NAME = 'red_fruit'


# ============================================================================
# Sub-node 1 — Arm controller & joint state subscriber
# ============================================================================

class _OctopusArmNode(Node):
    """
    Owns:
        • /joint_states subscriber
        • FollowJointTrajectory action client
        • /pause_physics, /unpause_physics, /reset_world service clients
    """

    def __init__(self):
        super().__init__('octopus_arm_rl_node')

        self._joint_positions  = np.zeros(NUM_JOINTS)
        self._joint_velocities = np.zeros(NUM_JOINTS)
        self._obs_lock         = threading.Lock()
        self._obs_event        = threading.Event()

        self.create_subscription(JointState, '/joint_states', self._joint_states_cb, 10)

        self._action_client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory',
        )

        self._pause_client   = self.create_client(Empty, '/pause_physics')
        self._unpause_client = self.create_client(Empty, '/unpause_physics')
        self._reset_client   = self.create_client(Empty, '/reset_world')

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _joint_states_cb(self, msg: JointState):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        positions   = np.zeros(NUM_JOINTS)
        velocities  = np.zeros(NUM_JOINTS)

        for i, joint in enumerate(JOINT_NAMES):
            if joint in name_to_idx:
                j = name_to_idx[joint]
                positions[i]  = msg.position[j] if msg.position else 0.0
                velocities[i] = msg.velocity[j] if msg.velocity else 0.0

        with self._obs_lock:
            self._joint_positions  = positions
            self._joint_velocities = velocities

        self._obs_event.set()

    # ── Getters ────────────────────────────────────────────────────────────

    def get_joint_obs(self) -> tuple[np.ndarray, np.ndarray]:
        with self._obs_lock:
            return self._joint_positions.copy(), self._joint_velocities.copy()

    # ── Gazebo control ─────────────────────────────────────────────────────

    def pause(self):
        if self._pause_client.service_is_ready():
            self._pause_client.call_async(Empty.Request())

    def unpause(self):
        if self._unpause_client.service_is_ready():
            self._unpause_client.call_async(Empty.Request())

    def reset_world(self):
        if self._reset_client.service_is_ready():
            future = self._reset_client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

    # ── Action ─────────────────────────────────────────────────────────────

    def send_joint_goal(self, target_positions: np.ndarray, duration_sec: float = TRAJECTORY_TIME):
        if not self._action_client.server_is_ready():
            self.get_logger().warn('Action server not ready, skipping goal.')
            return

        point                  = JointTrajectoryPoint()
        point.positions        = target_positions.tolist()
        point.velocities       = [0.0] * NUM_JOINTS
        point.time_from_start  = Duration(
            seconds=int(duration_sec),
            nanoseconds=int((duration_sec % 1) * 1e9),
        ).to_msg()

        goal                          = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names   = JOINT_NAMES
        goal.trajectory.points        = [point]

        self._action_client.send_goal_async(goal)   # fire-and-forget

    # ── Service readiness ──────────────────────────────────────────────────

    def wait_for_services(self, timeout_sec: float = 10.0):
        for client, name in [
            (self._pause_client,   '/pause_physics'),
            (self._unpause_client, '/unpause_physics'),
            (self._reset_client,   '/reset_world'),
        ]:
            deadline = time.time() + timeout_sec
            while not client.wait_for_service(timeout_sec=1.0):
                if time.time() > deadline:
                    self.get_logger().error(f'Service {name} unavailable after {timeout_sec}s')
                    break
                self.get_logger().info(f'Waiting for {name}…')

        self.get_logger().info('Waiting for action server…')
        self._action_client.wait_for_server(timeout_sec=timeout_sec)
        self.get_logger().info('All arm services ready.')


# ============================================================================
# Sub-node 2 — RGB-D camera object detector
# ============================================================================

class _ObjectDetectorNode(Node):
    """
    Subscribes to the RGB and depth cameras.
    Detects the red fruit by HSV colour segmentation and back-projects
    its pixel centroid to 3-D using the depth image + camera intrinsics.

    The 3-D point (in camera frame) is published on /detected_object_3d
    AND stored in self._detected_position for direct use by the RL env.

    The detected position is also used as part of the RL observation vector.
    """

    # Sentinel: returned when detection is invalid
    NO_DETECTION = np.zeros(3, dtype=np.float32)

    # HSV range for red (wraps around 0°/180° in OpenCV)
    _HSV_LOWER1 = np.array([0,   120,  70])
    _HSV_UPPER1 = np.array([10,  255, 255])
    _HSV_LOWER2 = np.array([170, 120,  70])
    _HSV_UPPER2 = np.array([180, 255, 255])

    def __init__(self, visualize: bool = False):
        super().__init__('object_detector_node')

        self._bridge      = CvBridge()
        self._visualize   = visualize

        # Camera intrinsics
        self._fx = self._fy = self._cx = self._cy = None

        # Latest depth image (numpy float32 in metres)
        self._depth_image: np.ndarray | None = None

        # Latest detection result (3,) in camera frame; zeros if no detection
        self._detected_position = self.NO_DETECTION.copy()
        self._detection_lock    = threading.Lock()
        self._detection_event   = threading.Event()

        # ── Subscribers ────────────────────────────────────────────────────
        self.create_subscription(
            Image, '/octopus/rgb_cam/image_raw', self._rgb_cb, 10)
        self.create_subscription(
            Image, '/octopus/depth_cam/depth/image_raw', self._depth_cb, 10)
        self.create_subscription(
            CameraInfo, '/octopus/depth_cam/depth/camera_info', self._info_cb, 10)

        # ── Publisher ──────────────────────────────────────────────────────
        self._pub = self.create_publisher(PointStamped, '/detected_object_3d', 10)

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _info_cb(self, msg: CameraInfo):
        self._fx = msg.k[0]
        self._fy = msg.k[4]
        self._cx = msg.k[2]
        self._cy = msg.k[5]

    def _depth_cb(self, msg: Image):
        self._depth_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _rgb_cb(self, msg: Image):
        if self._depth_image is None or self._fx is None:
            return

        frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red wraps around in HSV — combine two masks
        mask1 = cv2.inRange(hsv, self._HSV_LOWER1, self._HSV_UPPER1)
        mask2 = cv2.inRange(hsv, self._HSV_LOWER2, self._HSV_UPPER2)
        mask  = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            with self._detection_lock:
                self._detected_position = self.NO_DETECTION.copy()
            return

        cnt      = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        u = int(x + w / 2)
        v = int(y + h / 2)

        # Guard against out-of-bounds pixel access
        h_img, w_img = self._depth_image.shape[:2]
        u = np.clip(u, 0, w_img - 1)
        v = np.clip(v, 0, h_img - 1)

        Z = float(self._depth_image[v, u])
        if Z <= 0.0 or not np.isfinite(Z):
            with self._detection_lock:
                self._detected_position = self.NO_DETECTION.copy()
            return

        X = (u - self._cx) * Z / self._fx
        Y = (v - self._cy) * Z / self._fy

        # Clamp to sane range (avoid wild values during reset)
        pos = np.array([X, Y, Z], dtype=np.float32)
        pos = np.clip(pos, -50.0, 50.0)

        with self._detection_lock:
            self._detected_position = pos

        self._detection_event.set()

        # Publish for external subscribers
        pt_msg                  = PointStamped()
        pt_msg.header.stamp     = self.get_clock().now().to_msg()
        pt_msg.header.frame_id  = 'camera_link'
        pt_msg.point.x          = float(pos[0])
        pt_msg.point.y          = float(pos[1])
        pt_msg.point.z          = float(pos[2])
        self._pub.publish(pt_msg)

        self.get_logger().debug(
            f"Detected fruit @ camera frame: X={pos[0]:.2f} Y={pos[1]:.2f} Z={pos[2]:.2f}")

        if self._visualize:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)
            cv2.imshow('Detection', frame)
            cv2.waitKey(1)

    # ── Getter ─────────────────────────────────────────────────────────────

    def get_detected_position(self) -> np.ndarray:
        """Returns (3,) float32 position in camera frame, or zeros if no detection."""
        with self._detection_lock:
            return self._detected_position.copy()


# ============================================================================
# Sub-node 3 — Gazebo ground-truth fruit position (for reward)
# ============================================================================

class _FruitPositionNode(Node):
    """
    Reads /gazebo/model_states to get the ground-truth world position of
    the 'red_fruit' model.

    This is used ONLY for reward calculation.  The RL policy receives the
    camera-detected position (noisy/vision-based), not this ground truth.

    Also re-publishes to /fruit_world_position and /fruit_target_position
    so external tools (RViz, rosbag) can record them.
    """

    def __init__(self):
        super().__init__('fruit_position_node')

        gazebo_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            ModelStates, '/gazebo/model_states', self._model_states_cb, gazebo_qos)

        self._fruit_pub  = self.create_publisher(PointStamped, '/fruit_world_position',  10)
        self._target_pub = self.create_publisher(PointStamped, '/fruit_target_position', 10)

        self._fruit_position = FRUIT_START_POSITION.copy()
        self._pos_lock       = threading.Lock()
        self._fruit_found    = False

        # Periodically publish static target so the env can always subscribe to it
        self.create_timer(0.1, self._publish_target)

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _model_states_cb(self, msg: ModelStates):
        if FRUIT_MODEL_NAME not in msg.name:
            if not self._fruit_found:
                self.get_logger().warn_once(
                    f"'{FRUIT_MODEL_NAME}' not in /gazebo/model_states — "
                    "check your SDF model name."
                )
            return

        self._fruit_found = True
        idx  = msg.name.index(FRUIT_MODEL_NAME)
        pose = msg.pose[idx]

        with self._pos_lock:
            self._fruit_position = np.array(
                [pose.position.x, pose.position.y, pose.position.z],
                dtype=np.float64,
            )

        # Publish world-frame fruit position for logging/RViz
        pt               = PointStamped()
        pt.header.stamp  = self.get_clock().now().to_msg()
        pt.header.frame_id = 'world'
        pt.point.x       = float(self._fruit_position[0])
        pt.point.y       = float(self._fruit_position[1])
        pt.point.z       = float(self._fruit_position[2])
        self._fruit_pub.publish(pt)

    def _publish_target(self):
        pt               = PointStamped()
        pt.header.stamp  = self.get_clock().now().to_msg()
        pt.header.frame_id = 'world'
        pt.point.x       = float(FRUIT_TARGET_POSITION[0])
        pt.point.y       = float(FRUIT_TARGET_POSITION[1])
        pt.point.z       = float(FRUIT_TARGET_POSITION[2])
        self._target_pub.publish(pt)

    # ── Getter ─────────────────────────────────────────────────────────────

    def get_fruit_position(self) -> np.ndarray:
        with self._pos_lock:
            return self._fruit_position.copy()

    @staticmethod
    def get_target_position() -> np.ndarray:
        return FRUIT_TARGET_POSITION.copy()


# ============================================================================
# Gymnasium Environment
# ============================================================================

class OctopusArmEnv(gym.Env):
    """
    Gymnasium environment — Octopus Arm Fruit Transport Task.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Observation  (36-dim float32)                                      │
    │    [0  :10]  joint positions          (rad)                         │
    │    [10 :20]  joint velocities         (rad/s)                       │
    │    [20 :30]  target joint config      (rad)  — arm pose goal        │
    │    [30 :33]  camera-detected fruit    (m, camera frame)             │
    │    [33 :36]  fruit drop-target        (m, world frame)              │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Action  (10-dim float32)                                           │
    │    Desired joint positions in [JOINT_MIN, JOINT_MAX] radians        │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Reward  (fruit-transport shaped)                                   │
    │    r  = W_ARM_POSE  * -||q - q_target||²       arm pose penalty    │
    │       + W_FRUIT     * -||p_fruit - p_drop||²   fruit dist penalty  │
    │       - TIME_PENALTY                            speed encouragement │
    │       + BONUS_FRUIT   when fruit reaches drop zone                  │
    │       + BONUS_ARM     when arm also at goal pose (secondary bonus)  │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Termination                                                        │
    │    terminated  — fruit within DROP_TOLERANCE of target              │
    │                  AND arm within GOAL_TOLERANCE of joint target      │
    │    truncated   — step_count >= MAX_STEPS                            │
    └─────────────────────────────────────────────────────────────────────┘
    """

    metadata = {'render_modes': []}

    def __init__(self, render_mode=None, visualize_camera: bool = False):
        """
        Args:
            render_mode:       Gymnasium render mode (unused, Gazebo handles rendering).
            visualize_camera:  If True, show OpenCV detection window (debug only).
        """
        super().__init__()

        # ── Observation & action spaces ────────────────────────────────────
        #
        #   Joint positions  : [JOINT_MIN, JOINT_MAX]
        #   Joint velocities : [-50, 50]  rad/s
        #   Target joints    : [JOINT_MIN, JOINT_MAX]
        #   Camera detection : [-50, 50]  m  (zeros when fruit undetected)
        #   Drop target      : fixed point; bounded to world range [-10, 10] m

        OBS_DIM = NUM_JOINTS * 3 + 3 + 3   # 36

        obs_low = np.concatenate([
            np.full(NUM_JOINTS, JOINT_MIN,  dtype=np.float32),  # positions
            np.full(NUM_JOINTS, -50.0,      dtype=np.float32),  # velocities
            np.full(NUM_JOINTS, JOINT_MIN,  dtype=np.float32),  # target joints
            np.full(3,          -50.0,      dtype=np.float32),  # camera detection
            np.full(3,          -10.0,      dtype=np.float32),  # drop target
        ])
        obs_high = np.concatenate([
            np.full(NUM_JOINTS, JOINT_MAX,  dtype=np.float32),
            np.full(NUM_JOINTS,  50.0,      dtype=np.float32),
            np.full(NUM_JOINTS, JOINT_MAX,  dtype=np.float32),
            np.full(3,           50.0,      dtype=np.float32),
            np.full(3,           10.0,      dtype=np.float32),
        ])

        assert obs_low.shape[0] == OBS_DIM, "Obs bounds dim mismatch"

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.float32(-DELTA_MAX),
            high=np.float32(DELTA_MAX),
            shape=(NUM_JOINTS,),
            dtype=np.float32,
        )

        # ── Episode state ──────────────────────────────────────────────────
        self._step_count    = 0
        self._arm_target    = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._episode_count = 0

        # ── ROS 2 init ─────────────────────────────────────────────────────
        if not rclpy.ok():
            rclpy.init()

        self._arm_node      = _OctopusArmNode()
        self._detector_node = _ObjectDetectorNode(visualize=visualize_camera)
        self._fruit_node    = _FruitPositionNode()

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._arm_node)
        self._executor.add_node(self._detector_node)
        self._executor.add_node(self._fruit_node)

        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        # Block until Gazebo services are alive
        self._arm_node.wait_for_services()

        # Unpause so topics start flowing, then wait for first joint observation
        self._arm_node.unpause()
        self._arm_node._obs_event.wait(timeout=5.0)

        self._arm_node.get_logger().info(
            f"OctopusArmEnv ready.  "
            f"Obs space: {self.observation_space.shape}  "
            f"Act space: {self.action_space.shape}"
        )

    # =========================================================================
    # Gymnasium API
    # =========================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_count += 1
        self._step_count = 0

        # 1. Pause sim
        self._arm_node.pause()
        time.sleep(0.05)

        # 2. Reset world — returns arm and fruit to SDF default poses
        self._arm_node.reset_world()
        time.sleep(RESET_SETTLE)

        # 3. Sample a new random arm target configuration
        if options and 'arm_target' in options:
            self._arm_target = np.array(options['arm_target'], dtype=np.float32)

        else:
            self._arm_target = self.np_random.uniform(
                low=JOINT_MIN*0.5,
                high=JOINT_MAX*0.5,
                size=(NUM_JOINTS,),
            ).astype(np.float32)
            
        self._current_cmd = np.zeros(NUM_JOINTS, dtype=np.float32)

        
        # 4. Unpause and collect a fresh observation
        self._arm_node._obs_event.clear()
        self._arm_node.unpause()
        self._arm_node._obs_event.wait(timeout=3.0)

        obs  = self._get_obs()
        info = {
            'episode':          self._episode_count,
            'arm_target':       self._arm_target.tolist(),
            'fruit_target':     FRUIT_TARGET_POSITION.tolist(),
            'fruit_position':   self._fruit_node.get_fruit_position().tolist(),
        }
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, JOINT_MIN, JOINT_MAX).astype(np.float32)
        # delta = np.clip(action, -DELTA_MAX, DELTA_MAX).astype(np.float32)
        # self._current_cmd = np.clip(
        #     self._current_cmd + delta,
        #     JOINT_MIN,
        #     JOINT_MAX,
        # ).astype(np.float32)
        
        # 1. Unpause sim
        self._arm_node.unpause()

        # 2. Send joint trajectory goal
        self._arm_node.send_joint_goal(action, duration_sec=TRAJECTORY_TIME)

        # 3. Wait for arm to move, collecting observations
        self._arm_node._obs_event.clear()
        time.sleep(STEP_DURATION)
        self._arm_node._obs_event.wait(timeout=5.0)

        # 4. Pause sim
        self._arm_node.pause()

        # 5. Gather state
        joint_positions, joint_velocities = self._arm_node.get_joint_obs()
        fruit_world_pos   = self._fruit_node.get_fruit_position()   # ground truth
        detected_cam_pos  = self._detector_node.get_detected_position()  # vision

        # self._current_cmd = joint_positions.copy()
        
        # 6. Compute reward & termination
        obs      = self._build_obs(joint_positions, joint_velocities, detected_cam_pos)
        reward   = self._compute_reward(joint_positions, fruit_world_pos)
        success  = self._check_success(joint_positions, fruit_world_pos)

        self._step_count += 1
        terminated = bool(success)
        truncated  = bool(self._step_count >= MAX_STEPS)

        # Distance of fruit to drop target (handy for logging / curriculum)
        fruit_dist = float(np.linalg.norm(fruit_world_pos - FRUIT_TARGET_POSITION))
        arm_dist   = float(np.linalg.norm(joint_positions - self._arm_target))

        info = {
            'step':              self._step_count,
            'success':           success,
            'arm_distance':      arm_dist,
            'fruit_distance':    fruit_dist,
            'joint_positions':   joint_positions.tolist(),
            'arm_target':        self._arm_target.tolist(),
            'fruit_world_pos':   fruit_world_pos.tolist(),
            'fruit_target':      FRUIT_TARGET_POSITION.tolist(),
            'detected_cam_pos':  detected_cam_pos.tolist(),
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        self._arm_node.unpause()   # leave sim running on exit
        self._executor.shutdown()
        self._arm_node.destroy_node()
        self._detector_node.destroy_node()
        self._fruit_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    # =========================================================================
    # Observation builder
    # =========================================================================

    def _get_obs(self) -> np.ndarray:
        joint_pos, joint_vel = self._arm_node.get_joint_obs()
        detected_cam         = self._detector_node.get_detected_position()
        return self._build_obs(joint_pos, joint_vel, detected_cam)

    def _build_obs(
        self,
        joint_positions:  np.ndarray,   # (10,) rad
        joint_velocities: np.ndarray,   # (10,) rad/s
        detected_cam_pos: np.ndarray,   # (3,)  m  in camera frame
    ) -> np.ndarray:
        """
        Assembles the 36-dim observation vector.

        Note: the drop-target is expressed in world frame (fixed), while the
        camera detection is in camera frame.  The policy must learn the
        implicit transformation — or you can add a TF lookup here if you
        prefer both in world frame.
        """
        return np.concatenate([
            joint_positions.astype(np.float32),               # [0:10]
            joint_velocities.astype(np.float32),              # [10:20]
            self._arm_target.astype(np.float32),              # [20:30]
            detected_cam_pos.astype(np.float32),              # [30:33]
            FRUIT_TARGET_POSITION.astype(np.float32),         # [33:36]
        ])

    # =========================================================================
    # Reward & termination
    # =========================================================================

    def _compute_reward(
        self,
        joint_positions: np.ndarray,    # (10,) current arm config
        fruit_world_pos: np.ndarray,    # (3,)  fruit in world frame
    ) -> float:
        """
        Fruit-transport shaped reward:

            r = W_ARM_POSE * -||q - q_target||²          ← encourage correct arm pose
              + W_FRUIT    * -||p_fruit - p_drop||²       ← encourage fruit delivery
              - TIME_PENALTY                              ← encourage speed
              + BONUS_FRUIT   (if fruit in drop zone)
              + BONUS_ARM     (if arm also at target, secondary)

        The fruit term dominates (W_FRUIT > W_ARM_POSE) so the policy
        prioritises moving the fruit over arm pose accuracy.
        """
        # ── Distance penalties ────────────────────────────────────────────
        arm_dist_sq   = float(np.sum((joint_positions - self._arm_target) ** 2))
        fruit_dist    = float(np.linalg.norm(fruit_world_pos - FRUIT_TARGET_POSITION))
        fruit_dist_sq = fruit_dist ** 2

        reward  = W_ARM_POSE * (-arm_dist_sq)
        reward += W_FRUIT    * (-fruit_dist_sq)
        reward -= TIME_PENALTY

        # ── Proximity shaping: extra dense signal as fruit nears target ───
        # Exponential bonus ramps up continuously as fruit approaches goal.
        # This fills the sparse-reward gap without needing the fruit to
        # actually reach the target before receiving positive feedback.
        max_dist  = float(np.linalg.norm(
            FRUIT_START_POSITION - FRUIT_TARGET_POSITION))  # normalise
        if max_dist > 1e-6:
            proximity_bonus = BONUS_FRUIT * 0.3 * np.exp(-3.0 * fruit_dist / max_dist)
            reward += float(proximity_bonus)

        # ── Success bonuses ───────────────────────────────────────────────
        fruit_success = fruit_dist < DROP_TOLERANCE
        arm_success   = bool(np.all(np.abs(joint_positions - self._arm_target) < GOAL_TOLERANCE))

        if fruit_success:
            reward += BONUS_FRUIT        # primary delivery bonus
        if fruit_success and arm_success:
            reward += BONUS_ARM          # secondary bonus: arm also at goal

        return reward

    def _check_success(
        self,
        joint_positions: np.ndarray,
        fruit_world_pos: np.ndarray,
    ) -> bool:
        """
        Episode succeeds when the fruit has been delivered to the drop zone
        AND the arm has reached its target configuration.
        """
        fruit_delivered = float(np.linalg.norm(
            fruit_world_pos - FRUIT_TARGET_POSITION)) < DROP_TOLERANCE
        arm_at_goal     = bool(np.all(
            np.abs(joint_positions - self._arm_target) < GOAL_TOLERANCE))
        return fruit_delivered and arm_at_goal


# ============================================================================
# Quick sanity-check demo
# ============================================================================

def _demo():
    print("=" * 65)
    print("Octopus Arm RL Environment v2 — Fruit Transport Demo")
    print("Make sure Gazebo is running first!")
    print("=" * 65)

    env = OctopusArmEnv(visualize_camera=True)
    print(f"\nObservation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}")
    print(f"\nObs layout:")
    print("  [0:10]  joint positions  (rad)")
    print("  [10:20] joint velocities (rad/s)")
    print("  [20:30] target joints    (rad)")
    print("  [30:33] camera fruit pos (m, camera frame)")
    print("  [33:36] drop target pos  (m, world frame)")

    for ep in range(2):
        obs, info = env.reset()
        print(f"\n── Episode {ep + 1} {'─' * 48}")
        print(f"  Arm target  : {[f'{v:.3f}' for v in info['arm_target']]}")
        print(f"  Fruit target: {info['fruit_target']}")
        print(f"  Fruit start : {info['fruit_position']}")

        total_reward = 0.0
        for step in range(MAX_STEPS):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if step % 30 == 0:
                print(
                    f"  Step {step:3d} | r={reward:8.2f} | "
                    f"arm_dist={info['arm_distance']:.3f} | "
                    f"fruit_dist={info['fruit_distance']:.3f} | "
                    f"success={info['success']}"
                )

            if terminated or truncated:
                reason = 'SUCCESS 🎉' if terminated else 'TIMEOUT'
                print(f"  → Episode ended ({reason}) at step {step + 1}")
                break

        print(f"  Total reward: {total_reward:.2f}")

    env.close()
    print("\nDemo complete.")


# ============================================================================
# Stable-Baselines3 training entry point (optional)
# ============================================================================

def _train_sb3(total_timesteps: int = 2000):
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("stable-baselines3 not installed. Run:  pip install stable-baselines3")
        return

    env = OctopusArmEnv()

    print("Running Gymnasium env checker…")
    check_env(env, warn=True)

    print(f"\nStarting SAC training for {total_timesteps} timesteps…")
    model = SAC(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=2000,
        batch_size=4,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        tensorboard_log='./octopus_arm_tb/',
    )

    #load model if file exists
    import os
    if os.path.exists('/octopus_arm_fruit_sac.zip'):
        print("Loading existing model from octopus_arm_fruit_sac.zip")
        model = SAC.load('/octopus_arm_fruit_sac', env=env)
    model.learn(total_timesteps=total_timesteps)
    model.save('/octopus_arm_fruit_sac')
    print('Model saved → /octopus_arm_fruit_sac.zip')
    env.close()


# ============================================================================
# Entry point
# ============================================================================

def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        _train_sb3()
    else:
        _demo()


if __name__ == '__main__':
    main()