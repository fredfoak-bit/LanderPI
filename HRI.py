#!/usr/bin/env python3
# encoding: utf-8
import cv2
import time
import rclpy
import queue
import threading
import numpy as np
import mediapipe as mp
import os
from rclpy.node import Node
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from servo_controller_msgs.msg import ServosPosition
from servo_controller.bus_servo_control import set_servo_position

# --- IMPORTS FOR DIRECT SPEECH ---
from speech import speech
from large_models.config import * # ---------------------------------

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

class FistStopNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)
        self.name = name
        
        # 1. Initialize Speech
        self.language = os.environ.get("ASR_LANGUAGE", "English")
        try:
            if self.language == 'Chinese':
                self.tts_engine = speech.RealTimeTTS(log=self.get_logger())
            else:
                self.tts_engine = speech.RealTimeOpenAITTS(log=self.get_logger())
            self.get_logger().info("Speech Engine Ready!")
        except Exception as e:
            self.get_logger().error(f"Failed to init speech engine: {e}")
            self.tts_engine = None

        # 2. Initialize Hand Detector
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5
        )
        self.drawing = mp.solutions.drawing_utils
        
        # 3. Publishers
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1) 
        # Correct Publisher for Servos
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)

        # 4. Camera Subscription
        self.camera_topic = '/ascamera/camera_publisher/rgb0/image'
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=2)
        from sensor_msgs.msg import Image
        self.create_subscription(Image, self.camera_topic, self.image_callback, 1)

        # 5. State Flags
        self.running = True
        self.fist_detected = False # Shared flag between threads
        
        # 6. Start Threads
        # Thread 1: Process Images (Fast)
        threading.Thread(target=self.image_proc, daemon=True).start()
        # Thread 2: Control Robot Behavior (Slow Sequence)
        threading.Thread(target=self.control_loop, daemon=True).start()
        
        self.get_logger().info('Fist Stop Node Started.')

    def image_callback(self, ros_image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
            rgb_image = np.array(cv_image, dtype=np.uint8)
            if self.image_queue.full():
                self.image_queue.get()
            self.image_queue.put(rgb_image)
        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")

    def is_fist(self, landmarks, shape):
        h, w, _ = shape
        def get_coord(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

        wrist = get_coord(WRIST)
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
            if np.linalg.norm(tip - wrist) < np.linalg.norm(mcp - wrist) * 1.2: 
                folded_count += 1
        return folded_count >= 3

    def move_camera(self, pitch_angle):
        """
        pitch_angle: 500 is roughly horizon, 750 is looking down, 300 is looking up
        """
        # Servo 1: Pan (500 center), Servo 2: Tilt (pitch_angle)
        # Duration: 1.0 seconds
        set_servo_position(self.joints_pub, 1.0, ((1, 500), (2, pitch_angle)))
        time.sleep(1.0) # Wait for servo to move

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        self.mecanum_pub.publish(twist)

    def move_forward(self, speed=0.2):
        twist = Twist()
        twist.linear.x = speed
        self.mecanum_pub.publish(twist)

    def speak(self, text):
        if self.tts_engine:
            try:
                self.tts_engine.tts(text, model=tts_model, voice=voice_model)
            except Exception as e:
                self.get_logger().error(f"Speech error: {e}")

    def control_loop(self):
        """
        Main logic loop: Forward -> Stop -> Look Up -> Check Fist -> Repeat/Stop
        """
        # Initial wait for system to settle
        time.sleep(2)
        
        # Reset Camera to look down/forward initially (750)
        self.move_camera(750) 
        
        while self.running:
            # 1. Move Forward
            self.get_logger().info("Moving Forward...")
            self.move_forward(0.2)
            time.sleep(3.0) # Move for 3 seconds
            
            # 2. Stop Moving
            self.get_logger().info("Stopping...")
            self.stop_robot()
            time.sleep(0.5) # Wait for complete stop
            
            # 3. Look Up
            self.get_logger().info("Looking Up...")
            self.move_camera(500) # 500 is higher than 750 (Horizon/Up)
            
            # 4. Check for Fist (Wait a moment to detect)
            time.sleep(1.0) # Give camera time to see
            
            if self.fist_detected:
                self.get_logger().warn("FIST SEEN! Stopping Permanently.")
                self.speak("Danger")
                self.stop_robot()
                self.running = False # Break the loop
                break
            else:
                self.get_logger().info("No Fist. Continuing...")
                # 5. Look Down/Reset
                self.move_camera(750) # Look back down/forward
                # Loop repeats
                
        self.stop_robot()
        rclpy.shutdown()

    def image_proc(self):
        """
        Fast loop for detection only
        """
        while self.running:
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            image_flip = cv2.flip(image, 1)
            bgr_image = cv2.cvtColor(image_flip, cv2.COLOR_RGB2BGR)
            results = self.hand_detector.process(image_flip)
            
            detected_now = False
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.drawing.draw_landmarks(
                        bgr_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    if self.is_fist(hand_landmarks.landmark, image_flip.shape):
                        detected_now = True
                        cv2.putText(bgr_image, "FIST DETECTED", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Update shared flag
            self.fist_detected = detected_now
            
            cv2.imshow(self.name, bgr_image)
            key = cv2.waitKey(1)
            if key == 27:
                self.running = False

def main():
    # Use the same node name as your setup.py entry point
    node = FistStopNode('fist_back_node')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == "__main__":
    main()