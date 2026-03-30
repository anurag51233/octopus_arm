"""
Octopus Arm RL Environment
===========================
Author : David Valencia
Date   : 2026

The installed SB3's Monitor inherits from gymnasium.Env and asserts
isinstance(env, gymnasium.Env), so we must inherit from gymnasium.Env.
gymnasium IS already installed (SB3 pulled it in as a dependency).

Only packages needed beyond ROS Foxy:
    pip install stable-baselines3 opencv-python
    (gymnasium is installed automatically as an SB3 dependency)

Run:
    python3 octopus_arm_rl_env.py --mode train --timesteps 300000
    python3 octopus_arm_rl_env.py --mode train --load checkpoints/octopus_sac_100000_steps
    python3 octopus_arm_rl_env.py --mode eval  --load checkpoints/best/best_model

Observation (dim=30):
    [ 0:10]  joint positions  (rad)
    [10:20]  joint velocities (rad/s)
    [20:23]  fruit world XYZ  (m)
    [23:26]  target world XYZ (m)
    [26:29]  detected obj XYZ (m, camera frame)

Action (dim=10):
    Incremental Δθ ∈ [-1, 1], scaled by MAX_DELTA_RAD before applying.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ── ROS 2 ──────────────────────────────────────────────────────────────────
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image, JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from std_srvs.srv import Empty

# ── Vision ─────────────────────────────────────────────────────────────────
from cv_bridge import CvBridge
import cv2

# ── RL ─────────────────────────────────────────────────────────────────────
# gymnasium IS present — SB3 installed it. Monitor asserts isinstance(env, gymnasium.Env)
import gymnasium
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gazebo_msgs.srv import SetEntityState

from rclpy.callback_groups import ReentrantCallbackGroup

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

JOINT_NAMES = [
    "Plane_002_joint", "Plane_003_joint", "Plane_004_joint",
    "Plane_005_joint", "Plane_006_joint", "Plane_007_joint",
    "Plane_008_joint", "Plane_009_joint", "Plane_010_joint",
    "Plane_011_joint",
]
NUM_JOINTS = len(JOINT_NAMES)

JOINT_LOWER = np.full(NUM_JOINTS, -math.pi, dtype=np.float32)
JOINT_UPPER = np.full(NUM_JOINTS,  math.pi, dtype=np.float32)

MAX_DELTA_RAD = 0.15          # max radians per step per joint (~8.6 deg)

FRUIT_INITIAL_POS = np.array([ 3.0, 0.0, 2.5], dtype=np.float32)
FRUIT_TARGET_POS  = np.array([-3.0, 0.0, 1.0], dtype=np.float32)

SUCCESS_THRESH      = 0.10    # metres
SUCCESS_BONUS       = 200.0
STEP_PENALTY        = 0.01
JOINT_LIMIT_PENALTY = 1.0
DIST_SCALE          = 5.0
MAX_EPISODE_STEPS   = 20

ACTION_SERVER    = "/joint_trajectory_controller/follow_joint_trajectory"
JOINT_STATE_TOP  = "/joint_states"
FRUIT_POS_TOP    = "/fruit_world_position"
FRUIT_TGT_TOP    = "/fruit_target_position"
DETECTED_OBJ_TOP = "/detected_object_3d"

TRAJ_EXEC_TIME   = 0.4       # seconds to wait after sending goal

OBS_DIM = NUM_JOINTS * 2 + 3 + 3 + 3   # = 30


# ═══════════════════════════════════════════════════════════════════════════
# ROS 2 node
# ═══════════════════════════════════════════════════════════════════════════

class OctopusRosNode(Node):
    """All ROS 2 I/O in one node, spun in a background daemon thread."""

    def __init__(self):
        super().__init__("octopus_rl_node")

        self._joint_lock = threading.Lock()
        self._fruit_lock = threading.Lock()
        self._det_lock   = threading.Lock()
        self._cam_lock   = threading.Lock()
        self._vis_frame   = None
        # Joint state
        self._joint_pos = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._joint_vel = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._last_js_time = time.time()

        # Fruit ground truth
        self._fruit_pos    = FRUIT_INITIAL_POS.copy()
        self._fruit_target = FRUIT_TARGET_POS.copy()
        
        self.cb_group = ReentrantCallbackGroup()
        
        self.create_subscription(
            JointState, JOINT_STATE_TOP, self._js_cb, 10, 
            callback_group=self.cb_group)

        self._action_client = ActionClient(
            self, FollowJointTrajectory, ACTION_SERVER, 
            callback_group=self.cb_group)

        self._reset_client = self.create_client(
            SetEntityState, "/gazebo/set_entity_state", 
            callback_group=self.cb_group)
        
        
        # Camera / detection
        self._bridge       = CvBridge()
        self._depth_image  = None
        self._cam_fx = self._cam_fy = self._cam_cx = self._cam_cy = None
        self._detected_pos = np.zeros(3, dtype=np.float32)

        self.create_subscription(
            CameraInfo, "/octopus/depth_cam/depth/camera_info",
            self._caminfo_cb, 10)
        self.create_subscription(
            Image, "/octopus/depth_cam/depth/image_raw",
            self._depth_cb, 10)
        self.create_subscription(
            Image, "/octopus/rgb_cam/image_raw",
            self._rgb_cb, 10)

        self._det_pub = self.create_publisher(PointStamped, DETECTED_OBJ_TOP, 10)

        # Trajectory action client
        self._action_client = ActionClient(
            self, FollowJointTrajectory, ACTION_SERVER)

        # Gazebo reset service
        self._reset_client = self.create_client(SetEntityState, "/gazebo/set_entity_state")

        self.get_logger().info("OctopusRosNode ready.")
        
    # ── Joint state ────────────────────────────────────────────────────────
    
    def _viz_loop(self):
        """Runs in its own thread. Only this thread ever calls cv2.imshow."""
        while rclpy.ok():
            with self._cam_lock:
                frame = self._vis_frame   # grab reference under lock
            
            if frame is not None:
                cv2.imshow("Fruit Detection", frame)
                cv2.waitKey(1)        # if this stalls, only viz thread is affected
            
            time.sleep(0.033)         # ~30 fps, no need to show every camera frame
        
        
    def _js_cb(self, msg: JointState):
        self._last_js_time = time.time()
        with self._joint_lock:
            for i, name in enumerate(JOINT_NAMES):
                if name in msg.name:
                    idx = msg.name.index(name)
                    self._joint_pos[i] = float(msg.position[idx])
                    if idx < len(msg.velocity):
                        self._joint_vel[i] = float(msg.velocity[idx])

    def get_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        age = time.time() - self._last_js_time
        if age > 1.0:
            print(f"[WARN] Joint state is STALE — last received {age:.1f}s ago", flush=True)
        with self._joint_lock:
            return self._joint_pos.copy(), self._joint_vel.copy()


    # ── Fruit / target ─────────────────────────────────────────────────────

    def _fruit_cb(self, msg: PointStamped):
        with self._fruit_lock:
            self._fruit_pos[:] = [msg.point.x, msg.point.y, msg.point.z]

    def _target_cb(self, msg: PointStamped):
        with self._fruit_lock:
            self._fruit_target[:] = [msg.point.x, msg.point.y, msg.point.z]

    def get_fruit_state(self) -> Tuple[np.ndarray, np.ndarray]:
        # print(f"DEBUG: get_fruit_state returning fruit at {self._fruit_pos[0]}") # Add this
        with self._fruit_lock:
            # print(f"DEBUG Inside: get_fruit_state returning fruit at {self._fruit_pos[0]}") # Add this
            
            return self._fruit_pos.copy(), self._fruit_target.copy()

    # ── Camera / detection ─────────────────────────────────────────────────

    def _caminfo_cb(self, msg: CameraInfo):
        with self._cam_lock:
            self._cam_fx = msg.k[0]
            self._cam_fy = msg.k[4]
            self._cam_cx = msg.k[2]
            self._cam_cy = msg.k[5]

    def _depth_cb(self, msg: Image):
        with self._cam_lock:
            self._depth_image = self._bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough")

    def _rgb_cb(self, msg: Image):
        with self._cam_lock:
            depth = self._depth_image
            fx, fy = self._cam_fx, self._cam_fy
            cx, cy = self._cam_cx, self._cam_cy
            

        if depth is None or fx is None:
            return

        frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red fruit — two HSV ranges cover hue wrap-around
        mask1 = cv2.inRange(hsv, np.array([0,   120,  70]),
                                 np.array([10,  255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 120,  70]),
                                 np.array([180, 255, 255]))
        mask  = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            with self._det_lock:                          # ← reset on no detection
                self._detected_pos[:] = [0.0, 0.0, 0.0]
            return


        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        u = int(x + w / 2)
        v = int(y + h / 2)

        if v >= depth.shape[0] or u >= depth.shape[1]:
            return

        Z = float(depth[v, u])
        if Z <= 0.0:
            return

        X = float(np.clip((u - cx) * Z / fx, -100.0, 100.0))
        Y = float(np.clip((v - cy) * Z / fy, -100.0, 100.0))
        Z = float(np.clip(Z, 0.0, 100.0))

        with self._det_lock:
            self._detected_pos[:] = [X, Y, Z]

        pt = PointStamped()
        pt.header.stamp    = self.get_clock().now().to_msg()
        pt.header.frame_id = "camera_link"
        pt.point.x, pt.point.y, pt.point.z = X, Y, Z
        self._det_pub.publish(pt)

        # ── Visualisation window ───────────────────────────────────────────
        vis = frame.copy()

        # Green bounding box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Red dot at centroid
        cv2.circle(vis, (u, v), 6, (0, 0, 255), -1)

        # Camera-frame 3-D label just above the box
        cam_label = f"X:{X:.2f}  Y:{Y:.2f}  Z:{Z:.2f} m"
        (tw, th), _ = cv2.getTextSize(
            cam_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lx = max(x, 0)
        ly = max(y - 8, th + 4)
        cv2.rectangle(vis, (lx, ly - th - 4), (lx + tw + 4, ly + 2),
                      (0, 0, 0), -1)
        cv2.putText(vis, cam_label, (lx + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
                    cv2.LINE_AA)

        # World-frame fruit position + distance to target (top-left overlay)
        with self._fruit_lock:
            fp = self._fruit_pos.copy()
            tp = self._fruit_target.copy()
        dist_to_tgt = float(np.linalg.norm(fp - tp))
        world_label = (f"Fruit world: ({fp[0]:.2f},{fp[1]:.2f},{fp[2]:.2f})"
                       f"  dist->target: {dist_to_tgt:.3f} m")
        cv2.putText(vis, world_label, (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 1,
                    cv2.LINE_AA)

        
        self._vis_frame = vis
        
        

    def get_detected_pos(self) -> np.ndarray:
        with self._det_lock:
            return self._detected_pos.copy()

    # ── Trajectory ─────────────────────────────────────────────────────────

    def send_joint_goal(self, target_positions: np.ndarray,
                        exec_time: float = TRAJ_EXEC_TIME) -> bool:
        if not self._action_client.wait_for_server(timeout_sec=2.0):
            return False

        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in target_positions]
        pt.time_from_start = Duration(
            seconds=0, nanoseconds=int(exec_time * 1e9)).to_msg()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        goal.trajectory.points = [pt]

        future = self._action_client.send_goal_async(goal)

        # Block until goal is accepted
        # rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        timeout = time.time() + 3.0
        while not future.done() and time.time() < timeout:
            time.sleep(0.01) # Let the background spin thread do the work
            
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            print("[ERROR] Goal rejected!", flush=True)
            return False

        print("[DEBUG] Goal accepted by action server", flush=True)

        # Block until execution completes
        result_future = goal_handle.get_result_async()
        
        timeout = time.time() + exec_time + 2.0
        while not result_future.done() and time.time() < timeout:
            time.sleep(0.01)
        return True

    # ── Gazebo reset ───────────────────────────────────────────────────────

    def reset_simulation(self):
        req = SetEntityState.Request()
        req.state.name = "red_fruit" # Match the name in your XML
        req.state.pose.position.x = 3.0
        req.state.pose.position.y = 2.0
        req.state.pose.position.z = 1.5
        # Reset velocity so it doesn't keep moving from the previous episode
        req.state.twist.linear.x = 0.0
        req.state.twist.linear.y = 0.0
        req.state.twist.linear.z = 0.0
        
        future = self._reset_client.call_async(req)
        # Wait for it to finish before proceeding
        timeout = time.time() + 3.0
        while not future.done() and time.time() < timeout:
            time.sleep(0.05)


# ═══════════════════════════════════════════════════════════════════════════
# Gymnasium environment
# Must inherit from gymnasium.Env because the installed SB3's Monitor
# does:  assert isinstance(env, gymnasium.Env)
# ═══════════════════════════════════════════════════════════════════════════

class OctopusArmEnv(gymnasium.Env):
    """
    Gymnasium env for the octopus arm fruit-delivery task.

    reset() -> (obs, info)           ← gymnasium API (5-tuple step)
    step()  -> (obs, reward, terminated, truncated, info)
    """

    metadata = {"render_modes": []}

    def __init__(self, ros_node: OctopusRosNode):
        super().__init__()
        self._node = ros_node

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(OBS_DIM,), dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(NUM_JOINTS,), dtype=np.float32)

        self._current_joint_pos = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._step_count        = 0
        self._prev_dist         = None

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        joint_pos, joint_vel = self._node.get_joint_state()
        fruit_pos, target    = self._node.get_fruit_state()
        detected             = self._node.get_detected_pos()
        return np.concatenate(
            [joint_pos, joint_vel, fruit_pos, target, detected]
        ).astype(np.float32)

    def _compute_reward(self, fruit_pos, target, joint_pos, success) -> float:
        dist = float(np.linalg.norm(fruit_pos - target))

        # Dense negative distance
        reward = -DIST_SCALE * dist

        # Potential-based shaping: positive when getting closer
        if self._prev_dist is not None:
            reward += DIST_SCALE * (self._prev_dist - dist)
        self._prev_dist = dist

        # Step cost
        reward -= STEP_PENALTY

        # Joint limit penalty
        n_viol = int(np.sum(
            (joint_pos < JOINT_LOWER) | (joint_pos > JOINT_UPPER)))
        reward -= JOINT_LIMIT_PENALTY * n_viol

        # Sparse success bonus
        if success:
            reward += SUCCESS_BONUS

        return float(reward)

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        """Returns (obs, info) as per gymnasium API."""
        super().reset(seed=seed)
        self._node.reset_simulation()
        time.sleep(1.0)

        home = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._node.send_joint_goal(home, exec_time=2.0)
        time.sleep(2.5)

        # Wait until joint states are fresh before proceeding
        deadline = time.time() + 5.0
        while time.time() - self._node._last_js_time > 0.5:
            if time.time() > deadline:
                print("[WARN] reset(): joint states still stale after 5s!", flush=True)
                break
            time.sleep(0.1)

        self._current_joint_pos, _ = self._node.get_joint_state()
        self._step_count = 0
        self._prev_dist  = None
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """Returns (obs, reward, terminated, truncated, info)."""
        self._step_count += 1

        # Scale normalised action to radians and clip to joint limits
        delta         = action.astype(np.float32) * MAX_DELTA_RAD
        new_joint_pos = np.clip(
            self._current_joint_pos + delta,
            JOINT_LOWER, JOINT_UPPER)

            # Snapshot BEFORE sending
        pos_before, _ = self._node.get_joint_state()

        
        accepted = self._node.send_joint_goal(new_joint_pos)
        if accepted:
            self._current_joint_pos = new_joint_pos.copy()

        # time.sleep(TRAJ_EXEC_TIME + 0.05)
        
        pos_after, _ = self._node.get_joint_state()
        actual_delta = np.abs(pos_after - pos_before)

        # ← THIS tells you if the arm actually moved
        print(f"[Step {self._step_count}] goal_accepted={accepted} "
            f"max_actual_delta={actual_delta.max():.4f} rad "
            f"max_intended_delta={np.abs(delta).max():.4f} rad "
            f"joints_moved={actual_delta.max() > 0.001}", flush=True)


        obs               = self._get_obs()
        fruit_pos, target = self._node.get_fruit_state()
        joint_pos, _      = self._node.get_joint_state()
        
        dist       = float(np.linalg.norm(fruit_pos - target))
        success    = dist < SUCCESS_THRESH
        terminated = success
        truncated  = self._step_count >= MAX_EPISODE_STEPS
        reward     = self._compute_reward(fruit_pos, target, joint_pos, success)

        print(f"[Step {self._step_count}] obs shape: {obs.shape}, "
              f"joint_pos: {joint_pos[:3]}, fruit_pos: {fruit_pos}, "
              f"target: {target}, detected: {obs[26:29]}, reward: {reward}", flush=True)
        
        info = {
            "distance_to_target": dist,
            "success":            success,
            "step":               self._step_count,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass   # Gazebo provides visualisation

    def close(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train(ros_node: OctopusRosNode,
          timesteps: int = 300_000,
          load_path: Optional[str] = None,
          checkpoint_dir: str = "checkpoints"):

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _make_env():
        env = OctopusArmEnv(ros_node)
        env = Monitor(env, filename=f"{checkpoint_dir}/monitor.csv")
        return env

    vec_env = DummyVecEnv([_make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if load_path:
        print(f"[RL] Loading model from: {load_path}")
        model = SAC.load(load_path, env=vec_env)
    else:
        model = SAC(
            policy          = "MlpPolicy",
            env             = vec_env,
            learning_rate   = 3e-4,
            buffer_size     = 100_000,
            learning_starts = 500,
            batch_size      = 256,
            tau             = 0.005,
            gamma           = 0.99,
            train_freq      = 1,
            gradient_steps  = 1,
            ent_coef        = "auto",
            policy_kwargs   = dict(net_arch=[256, 256]),
            verbose         = 1,
            tensorboard_log = "./tb_logs/",
        )

    checkpoint_cb = CheckpointCallback(
        save_freq         = 10_000,
        save_path         = checkpoint_dir,
        name_prefix       = "octopus_sac",
        save_vecnormalize = True,
    )

    eval_vec = DummyVecEnv([_make_env])
    eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    eval_cb = EvalCallback(
        eval_vec,
        best_model_save_path = f"{checkpoint_dir}/best",
        log_path             = f"{checkpoint_dir}/eval_logs",
        eval_freq            = 5_000,
        n_eval_episodes      = 3,
        deterministic        = True,
        verbose              = 1,
    )

    print(f"[RL] Starting SAC — {timesteps} steps ...")
    model.learn(
        total_timesteps = timesteps,
        callback        = [checkpoint_cb, eval_cb],
        progress_bar    = False,
    )

    model.save(f"{checkpoint_dir}/final_model")
    vec_env.save(f"{checkpoint_dir}/vec_normalize.pkl")
    print("[RL] Training complete.")


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(ros_node: OctopusRosNode,
             load_path: str,
             n_episodes: int = 10,
             vecnorm_path: Optional[str] = None):

    def _make_env():
        return OctopusArmEnv(ros_node)

    vec_env = DummyVecEnv([_make_env])
    if vecnorm_path:
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False

    model = SAC.load(load_path, env=vec_env)

    for ep in range(n_episodes):
        obs    = vec_env.reset()
        ep_ret = 0.0
        done   = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_ret += float(reward[0])
        print(f"Ep {ep+1:3d}  return={ep_ret:+8.2f}  "
              f"dist={info[0].get('distance_to_target', -1):.3f} m  "
              f"success={info[0].get('success', False)}")

    vec_env.close()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # import os
    # os.environ.setdefault("DISPLAY", ":0")   
    
    parser = argparse.ArgumentParser(
        description="Octopus Arm SAC — ROS 2 Foxy")
    parser.add_argument("--mode",           choices=["train", "eval"], default="train")
    parser.add_argument("--timesteps",      type=int, default=300_000)
    parser.add_argument("--load",           type=str, default=None)
    parser.add_argument("--vecnorm",        type=str, default=None)
    parser.add_argument("--episodes",       type=int, default=10)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    rclpy.init()
    ros_node = OctopusRosNode()
    
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(ros_node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    viz_thread = threading.Thread(
        target=ros_node._viz_loop, daemon=True)
    viz_thread.start()

    print("[ROS] Waiting 3 s for first sensor messages ...")
    time.sleep(3.0)

    try:
        if args.mode == "train":
            train(ros_node,
                  timesteps      = args.timesteps,
                  load_path      = args.load,
                  checkpoint_dir = args.checkpoint_dir)
        else:
            if not args.load:
                raise ValueError("--load <path> is required for eval mode")
            evaluate(ros_node,
                     load_path    = args.load,
                     n_episodes   = args.episodes,
                     vecnorm_path = args.vecnorm)
    except KeyboardInterrupt:
        print("\n[RL] Interrupted.")
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        print("[RL] Shutdown complete.")


if __name__ == "__main__":
    main()