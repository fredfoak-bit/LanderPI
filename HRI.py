#!/usr/bin/env python3
# encoding: utf-8
import cv2
import time
import math
import rclpy
import queue
import threading
import numpy as np
import mediapipe as mp
from rclpy.node import Node
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from ros_robot_controller_msgs.msg import BuzzerState

# MediaPipe constants
mp_hands = mp.solutions.hands
WRIST = mp_hands.HandLandmark.WRIST
THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP
INDEX_FINGER_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP
MIDDLE_FINGER_TIP = mp_hands.HandLandmark.MIDDLE_FINGER_TIP
RING_FINGER_TIP = mp_hands.HandLandmark.RING_FINGER_TIP
PINKY_TIP = mp_hands.HandLandmark.PINKY_TIP

INDEX_FINGER_MCP = mp_hands.HandLandmark.INDEX_FINGER_MCP
MIDDLE_FINGER_MCP = mp_hands.HandLandmark.MIDDLE_FINGER_MCP
RING_FINGER_MCP = mp_hands.HandLandmark.RING_FINGER_MCP
PINKY_MCP = mp_hands.HandLandmark.PINKY_MCP

class FistBackNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)
        self.name = name
        
        # Initialize Hand Detector
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5
        )
        self.drawing = mp.solutions.drawing_utils
        
        # Publishers
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1) # Chassis control
        self.buzzer_pub = self.create_publisher(BuzzerState, '/ros_robot_controller/set_buzzer', 1) # Warning sign
        
        # Camera Subscription
        self.camera_topic = '/ascamera/camera_publisher/rgb0/image'
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=2)
        self.create_subscription(self.get_msg_type_from_string('sensor_msgs/msg/Image'), self.camera_topic, self.image_callback, 1)

        self.running = True
        self.is_backing_up = False # Lock to prevent continuous triggering
        
        # Start processing thread
        threading.Thread(target=self.image_proc, daemon=True).start()
        self.get_logger().info('Fist Detection Node Started. Show a FIST to move back!')

    def get_msg_type_from_string(self, type_str):
        # dynamic import to avoid hardcoding issues if environments differ
        from sensor_msgs.msg import Image
        return Image

    def image_callback(self, ros_image):
        # Convert ROS image to OpenCV image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
            rgb_image = np.array(cv_image, dtype=np.uint8)
            if self.image_queue.full():
                self.image_queue.get()
            self.image_queue.put(rgb_image)
        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")

    def is_fist(self, landmarks, shape):
        """
        Detects if the hand is a closed fist.
        Logic: Distance from finger tips to wrist is significantly small.
        """
        h, w, _ = shape
        
        # Helper to get pixel coords
        def get_coord(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

        wrist = get_coord(WRIST)
        
        # Check if fingers are folded (Tip closer to wrist than MCP)
        # Using a simple distance threshold or comparison could work
        fingers_indices = [
            (INDEX_FINGER_TIP, INDEX_FINGER_MCP),
            (MIDDLE_FINGER_TIP, MIDDLE_FINGER_MCP),
            (RING_FINGER_TIP, RING_FINGER_MCP),
            (PINKY_TIP, PINKY_MCP)
        ]
        
        folded_count = 0
        for tip_idx, mcp_idx in fingers_indices:
            tip = get_coord(tip_idx)
            mcp = get_coord(mcp_idx)
            
            dist_tip_wrist = np.linalg.norm(tip - wrist)
            dist_mcp_wrist = np.linalg.norm(mcp - wrist)
            
            # If tip is closer to wrist than the knuckle (MCP), it's folded
            if dist_tip_wrist < dist_mcp_wrist * 1.2: # 1.2 is a tolerance factor
                folded_count += 1
                
        # If at least 3 fingers (excluding thumb) are folded, consider it a fist
        return folded_count >= 3

    def trigger_warning_and_move(self):
        if self.is_backing_up:
            return

        self.is_backing_up = True
        self.get_logger().warn("FIST DETECTED! Moving back...")

        # 1. Sound Buzzer
        buzzer_msg = BuzzerState()
        buzzer_msg.freq = 2000
        buzzer_msg.on_time = 0.5
        buzzer_msg.off_time = 0.1
        buzzer_msg.repeat = 2
        self.buzzer_pub.publish(buzzer_msg)

        # 2. Move Backwards
        twist = Twist()
        twist.linear.x = -0.2 # Move back at 0.2 m/s
        self.mecanum_pub.publish(twist)
        
        # Move for a short duration (e.g., 0.5 seconds -> approx 10cm)
        time.sleep(0.5) 
        
        # 3. Stop
        twist.linear.x = 0.0
        self.mecanum_pub.publish(twist)
        
        # Cooldown to prevent spamming
        time.sleep(1.0) 
        self.is_backing_up = False

    def image_proc(self):
        while self.running:
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            # Flip and process for display
            image_flip = cv2.flip(image, 1)
            bgr_image = cv2.cvtColor(image_flip, cv2.COLOR_RGB2BGR)
            
            results = self.hand_detector.process(image_flip)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.drawing.draw_landmarks(
                        bgr_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Check for Fist
                    if self.is_fist(hand_landmarks.landmark, image_flip.shape):
                        # Run action in a separate thread so visual loop doesn't freeze
                        threading.Thread(target=self.trigger_warning_and_move).start()
                        cv2.putText(bgr_image, "FIST: BACKING UP", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow(self.name, bgr_image)
            key = cv2.waitKey(1)
            if key == 27: # ESC
                self.running = False

        self.mecanum_pub.publish(Twist()) # Stop on exit
        rclpy.shutdown()

def main():
    node = FistBackNode('fist_back_node')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == "__main__":
    main()