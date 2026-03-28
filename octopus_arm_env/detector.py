import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge
import cv2
import numpy as np


class ObjectDetector(Node):

    def __init__(self):
        super().__init__('object_detector')

        self.bridge = CvBridge()

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            '/octopus/rgb_cam/image_raw',
            self.rgb_callback,
            10)

        self.depth_sub = self.create_subscription(
            Image,
            '/octopus/depth_cam/depth/image_raw',
            self.depth_callback,
            10)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/octopus/depth_cam/depth/camera_info',
            self.camera_info_callback,
            10)

        # Publisher
        self.publisher = self.create_publisher(
            PointStamped,
            '/detected_object_3d',
            10)

        # Storage
        self.depth_image = None
        self.fx = self.fy = self.cx = self.cy = None

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def rgb_callback(self, msg):

        if self.depth_image is None or self.fx is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # --- SIMPLE OBJECT DETECTION (color-based) ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 120, 70])
        upper = np.array([10, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return

        # Largest contour
        cnt = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(cnt)

        # Center pixel
        u = int(x + w / 2)
        v = int(y + h / 2)

        # Get depth
        Z = self.depth_image[v, u]

        if Z == 0:
            return
        raw_Z = self.depth_image[v, u]
        
        
        
        # Convert to meters if needed
        Z = float(Z)

        # --- 2D → 3D ---
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        # Publish
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = "camera_link"

        point_msg.point.x = X
        point_msg.point.y = Y
        point_msg.point.z = Z
        
        point_msg.point.x = np.clip(X, -2000, 2000)
        point_msg.point.y = np.clip(Y, -2000, 2000)
        point_msg.point.z = np.clip(Z, -2000, 2000)

        self.publisher.publish(point_msg)

        self.get_logger().info(f"3D Position: X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}")

        # Visualization
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.circle(frame, (u, v), 5, (0,0,255), -1)

        cv2.imshow("Detection", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()