import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import ApplyLinkWrench
from geometry_msgs.msg import Point, Wrench
import math

class WrenchFruitController(Node):
    def __init__(self):
        super().__init__('wrench_fruit_controller')
        
        self.fruit_name = 'red_fruit'
        self.link_name = 'red_fruit::link' # Note: model_name::link_name
        self.target_pos = [4.0, 0.0, 2.5]
        
        # PD Constants: Adjust these to change how "stiff" the air feels
        # High Kp = Stronger magnet, High Kd = More "thick air" (damping)
        self.kp = 50.0  
        self.kd = 5.0  

        self.client = self.create_client(ApplyLinkWrench, '/apply_link_wrench')
        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for Gazebo wrench service...')

        self.sub = self.create_subscription(
            ModelStates, '/gazebo/model_states', self.listener_callback, 10)
        
        self.get_logger().info('Wrench Controller Active: Fruit is now "buoyant".')

    def apply_wrench(self, fx, fy, fz):
        req = ApplyLinkWrench.Request()
        
        # TRY THIS: In many ROS 2 Gazebo setups, if the model name is unique, 
        # you just need the link name 'link' or the scoped name.
        req.link_name = 'red_fruit::link' 
        
        # Use an empty string to avoid the "Reference_frame" log warnings
        req.reference_frame = '' 

        wrench = Wrench()
        wrench.force.x = float(fx)
        wrench.force.y = float(fy)
        wrench.force.z = float(fz)
        req.wrench = wrench
        
        # Setting nanosec to 0 can sometimes help Gazebo apply it 
        # until the next physics update.
        req.duration.nanosec = 0 
        
        self.client.call_async(req)

    def listener_callback(self, msg):
        if self.fruit_name not in msg.name:
            return

        idx = msg.name.index(self.fruit_name)
        curr_pos = msg.pose[idx].position
        curr_vel = msg.twist[idx].linear

        dx = self.target_pos[0] - curr_pos.x
        dy = self.target_pos[1] - curr_pos.y
        dz = self.target_pos[2] - curr_pos.z

        # Increase Kp significantly to overcome any joint friction
        # Kp=100 is quite strong for a 0.1kg fruit
        kp = 100.0
        kd = 10.0

        force_x = (dx * kp) - (curr_vel.x * kd)
        force_y = (dy * kp) - (curr_vel.y * kd)
        force_z = (dz * kp) - (curr_vel.z * kd)

        # DEBUG: Add this log to see if the script thinks it's pushing
        if abs(dx) > 0.05:
            self.get_logger().info(f"Targeting {self.target_pos}. Current X: {curr_pos.x:.2f}. Pushing X with: {force_x:.2f}N")

        self.apply_wrench(force_x, force_y, force_z)
def main(args=None):
    rclpy.init(args=args)
    node = WrenchFruitController()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()