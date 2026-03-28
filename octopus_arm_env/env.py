"""
Octopus Arm Gazebo RL Environment
==================================
Author: Generated for David Valencia's octopus_arm_env package
Date:   2026

A Gymnasium-compatible environment wrapping the 10-joint octopus arm
simulated in Gazebo via ROS 2 Foxy.

Joints (10 total):
    Plane_002_joint ... Plane_011_joint

Action space:
    Box(10,) — desired joint positions in radians, clipped to [-pi, pi]

Observation space:
    Box(30,) — [joint_positions(10), joint_velocities(10), target_positions(10)]

Reward:
    - Shaped: negative L2 distance from current joint positions to target
    - Bonus:  +100 when all joints within tolerance of target
    - Penalty: -1 per step to encourage speed

Episode termination:
    - All joints within GOAL_TOLERANCE of target  (success)
    - Step count exceeds MAX_STEPS               (timeout)

Usage:
    # Make sure your Gazebo simulation is already running:
    #   ros2 launch octopus_arm_env octopus_arm_gazebo.launch.py
    #
    # Then in a separate terminal / script:
    #   python3 octopus_arm_env.py          # runs a quick random-action demo
    #
    # Or use with Stable-Baselines3:
    #   from octopus_arm_env import OctopusArmEnv
    #   from stable_baselines3 import SAC
    #   env = OctopusArmEnv()
    #   model = SAC("MlpPolicy", env, verbose=1)
    #   model.learn(total_timesteps=100_000)

Dependencies:
    pip install gymnasium numpy stable-baselines3   # optional SB3
    ROS 2 Foxy + gazebo_ros + control_msgs
"""

import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor

from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.duration import Duration

import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JOINT_NAMES = [
    'Plane_002_joint',
    'Plane_003_joint',
    'Plane_004_joint',
    'Plane_005_joint',
    'Plane_006_joint',
    'Plane_007_joint',
    'Plane_008_joint',
    'Plane_009_joint',
    'Plane_010_joint',
    'Plane_011_joint',
]
NUM_JOINTS      = len(JOINT_NAMES)

# Joint position limits (radians) — adjust to your robot's actual URDF limits
JOINT_MIN       = -np.pi/3.2
JOINT_MAX       =  np.pi/3.2

# How close each joint must be to the target to count as "success"
GOAL_TOLERANCE  = 0.05   # radians (~3 degrees)

# Maximum steps per episode before timeout
MAX_STEPS       = 200

# How long (wall-clock seconds) to wait after sending an action before
# reading the next observation.  Tune this to your controller update rate.
# With update_rate=100Hz you want at least 0.1s for the arm to respond.
STEP_DURATION   = 0.2    # seconds

# Time given to the trajectory controller to reach the target
TRAJECTORY_TIME = 1.0    # seconds

# How long reset() waits for the sim to settle after /reset_world
RESET_SETTLE    = 0.5    # seconds


# ---------------------------------------------------------------------------
# Internal ROS 2 Node
# ---------------------------------------------------------------------------

class _OctopusArmNode(Node):
    """
    Thin ROS 2 node that owns all pub/sub/action/service clients.
    Runs in a background thread via MultiThreadedExecutor so it never
    blocks the training loop.
    """

    def __init__(self):
        super().__init__('octopus_arm_rl_node')

        # ── Observation state ──────────────────────────────────────────────
        self._joint_positions = np.zeros(NUM_JOINTS)
        self._joint_velocities = np.zeros(NUM_JOINTS)
        self._obs_lock = threading.Lock()
        self._obs_event = threading.Event()   # fires every time /joint_states arrives

        # ── Joint state subscriber ─────────────────────────────────────────
        self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_states_cb,
            10
        )

        # ── Trajectory action client ───────────────────────────────────────
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )

        # ── Gazebo service clients ─────────────────────────────────────────
        self._pause_client   = self.create_client(Empty, '/pause_physics')
        self._unpause_client = self.create_client(Empty, '/unpause_physics')
        self._reset_client   = self.create_client(Empty, '/reset_world')

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _joint_states_cb(self, msg: JointState):
        """
        Map incoming /joint_states into ordered arrays matching JOINT_NAMES.
        The controller may publish joints in any order, so we index by name.
        """
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        positions  = np.zeros(NUM_JOINTS)
        velocities = np.zeros(NUM_JOINTS)

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

    def get_obs(self):
        with self._obs_lock:
            return (
                self._joint_positions.copy(),
                self._joint_velocities.copy()
            )

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
            # spin until done so we know reset completed before continuing
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

    # ── Action ─────────────────────────────────────────────────────────────

    def send_joint_goal(self, target_positions: np.ndarray, duration_sec: float = TRAJECTORY_TIME):
        """
        Send a FollowJointTrajectory goal asynchronously.
        Returns immediately — observation is read after STEP_DURATION sleep.
        """
        if not self._action_client.server_is_ready():
            self.get_logger().warn('Action server not ready, skipping goal.')
            return

        point = JointTrajectoryPoint()
        point.positions      = target_positions.tolist()
        point.velocities     = [0.0] * NUM_JOINTS
        point.time_from_start = Duration(
            seconds=int(duration_sec),
            nanoseconds=int((duration_sec % 1) * 1e9)
        ).to_msg()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        goal.trajectory.points      = [point]

        # fire-and-forget: we don't wait for the result here
        self._action_client.send_goal_async(goal)

    # ── Service readiness ──────────────────────────────────────────────────

    def wait_for_services(self, timeout_sec: float = 10.0):
        services = [
            (self._pause_client,   '/pause_physics'),
            (self._unpause_client, '/unpause_physics'),
            (self._reset_client,   '/reset_world'),
        ]
        for client, name in services:
            deadline = time.time() + timeout_sec
            while not client.wait_for_service(timeout_sec=1.0):
                if time.time() > deadline:
                    self.get_logger().error(f'Service {name} not available after {timeout_sec}s')
                    break
                self.get_logger().info(f'Waiting for {name}...')

        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server(timeout_sec=timeout_sec)
        self.get_logger().info('All services ready.')


# ---------------------------------------------------------------------------
# Gymnasium Environment
# ---------------------------------------------------------------------------

class OctopusArmEnv(gym.Env):
    """
    Gymnasium environment for the 10-joint octopus arm in Gazebo.

    Observation (30-dim):
        [0:10]  — current joint positions  (rad)
        [10:20] — current joint velocities (rad/s)
        [20:30] — target  joint positions  (rad)

    Action (10-dim):
        Desired joint positions (rad), clipped to [JOINT_MIN, JOINT_MAX].
        The action is sent as a single-waypoint FollowJointTrajectory goal.

    Reward:
        r = -||q_current - q_target||²   (dense distance penalty)
        +100 bonus when goal reached
        -1   per step (time penalty)
    """

    metadata = {'render_modes': []}

    def __init__(self, render_mode=None):
        super().__init__()

        # ── Spaces ────────────────────────────────────────────────────────
        obs_low  = np.array(
            [JOINT_MIN] * NUM_JOINTS +   # positions
            [-50.0]     * NUM_JOINTS +   # velocities
            [JOINT_MIN] * NUM_JOINTS,    # target positions
            dtype=np.float32
        )
        obs_high = np.array(
            [JOINT_MAX] * NUM_JOINTS +
            [50.0]      * NUM_JOINTS +
            [JOINT_MAX] * NUM_JOINTS,
            dtype=np.float32
        )

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(
            low=np.float32(JOINT_MIN),
            high=np.float32(JOINT_MAX),
            shape=(NUM_JOINTS,),
            dtype=np.float32
        )

        # ── Episode state ─────────────────────────────────────────────────
        self._step_count    = 0
        self._target        = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._episode_count = 0

        # ── ROS 2 init ────────────────────────────────────────────────────
        if not rclpy.ok():
            rclpy.init()

        self._node     = _OctopusArmNode()
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)

        # Spin executor in a background daemon thread
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            daemon=True
        )
        self._spin_thread.start()

        # Block here until Gazebo services are alive
        self._node.wait_for_services()

        # Unpause so /joint_states starts flowing, then wait for first obs
        self._node.unpause()
        self._node._obs_event.wait(timeout=5.0)

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_count += 1
        self._step_count = 0

        # 1. Pause sim
        self._node.pause()
        time.sleep(0.05)

        # 2. Reset world (returns arm to URDF default pose)
        self._node.reset_world()
        time.sleep(RESET_SETTLE)

        # 3. Sample a new random target
        if options and 'target' in options:
            # allow manual target override:  env.reset(options={'target': my_array})
            self._target = np.array(options['target'], dtype=np.float32)
        else:
            self._target = self.np_random.uniform(
                low=JOINT_MIN * 0.5,    # keep targets within ±π/2 by default
                high=JOINT_MAX * 0.5,
                size=(NUM_JOINTS,)
            ).astype(np.float32)

        # 4. Unpause and wait for a fresh observation
        self._node._obs_event.clear()
        self._node.unpause()
        self._node._obs_event.wait(timeout=3.0)

        obs  = self._get_obs()
        info = {'target': self._target.tolist(), 'episode': self._episode_count}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, JOINT_MIN, JOINT_MAX).astype(np.float32)

        # 1. Unpause sim
        self._node.unpause()

        # 2. Send action to trajectory controller
        self._node.send_joint_goal(action, duration_sec=TRAJECTORY_TIME)

        # 3. Wait STEP_DURATION for arm to respond, collecting observations
        self._node._obs_event.clear()
        time.sleep(STEP_DURATION)
        self._node._obs_event.wait(timeout=2.0)

        # 4. Pause sim
        self._node.pause()

        # 5. Compute reward, check termination
        positions, velocities = self._node.get_obs()
        obs     = self._build_obs(positions, velocities)
        reward  = self._compute_reward(positions)
        success = self._check_success(positions)

        self._step_count += 1
        terminated = bool(success)
        truncated  = bool(self._step_count >= MAX_STEPS)

        info = {
            'step':           self._step_count,
            'success':        success,
            'distance':       float(np.linalg.norm(positions - self._target)),
            'joint_positions': positions.tolist(),
            'target':         self._target.tolist(),
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        self._node.unpause()   # leave sim running on exit
        self._executor.shutdown()
        self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        positions, velocities = self._node.get_obs()
        return self._build_obs(positions, velocities)

    def _build_obs(self, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        return np.concatenate([
            positions.astype(np.float32),
            velocities.astype(np.float32),
            self._target.astype(np.float32),
        ])

    def _compute_reward(self, positions: np.ndarray) -> float:
        distance = np.linalg.norm(positions - self._target)
        reward   = -float(distance ** 2)   # dense penalty: closer = less negative
        reward  -= 1.0                     # time penalty
    
        if self._check_success(positions):
            reward += 100.0                # success bonus

        return reward

    def _check_success(self, positions: np.ndarray) -> bool:
        return bool(np.all(np.abs(positions - self._target) < GOAL_TOLERANCE))


# ---------------------------------------------------------------------------
# Quick sanity-check demo  (run this file directly to test)
# ---------------------------------------------------------------------------

def _demo():
    """
    Runs 2 episodes with random actions.
    Make sure Gazebo is already running before executing this.
    """
    print("=" * 60)
    print("Octopus Arm RL Environment — sanity check demo")
    print("Make sure Gazebo is running first!")
    print("=" * 60)

    env = OctopusArmEnv()
    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}\n")

    for episode in range(2):
        obs, info = env.reset()
        print(f"\n── Episode {episode + 1} ──────────────────────────")
        print(f"Target positions: {[f'{v:.3f}' for v in info['target']]}")

        total_reward = 0.0
        for step in range(MAX_STEPS):
            action          = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward   += reward

            if step % 20 == 0:
                print(f"  Step {step:3d} | reward={reward:7.2f} | "
                      f"dist={info['distance']:.4f} | success={info['success']}")

            if terminated or truncated:
                reason = "SUCCESS" if terminated else "TIMEOUT"
                print(f"  → Episode ended ({reason}) at step {step + 1}")
                break

        print(f"  Total reward: {total_reward:.2f}")

    env.close()
    print("\nDemo complete.")


# ---------------------------------------------------------------------------
# Stable-Baselines3 training entry point  (optional)
# ---------------------------------------------------------------------------

def _train_sb3(total_timesteps: int = 100_000):
    """
    Trains a SAC agent using Stable-Baselines3.
    Install with:  pip install stable-baselines3
    """
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        print("stable-baselines3 not installed. Run: pip install stable-baselines3")
        return

    env = OctopusArmEnv()

    print("Running environment checker...")
    check_env(env, warn=True)

    print(f"\nStarting SAC training for {total_timesteps} timesteps...")
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        tensorboard_log="./octopus_arm_tb/",
    )

    model.learn(total_timesteps=total_timesteps)
    model.save("octopus_arm_sac")
    print("Model saved to octopus_arm_sac.zip")

    env.close()


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        _train_sb3()
    else:
        _demo()