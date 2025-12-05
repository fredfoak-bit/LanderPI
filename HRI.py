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
import os
from rclpy.node import Node
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

# --- NEW IMPORTS FOR DIRECT SPEECH ---
from speech import speech
from large_models.config import * # Imports api_key, tts_model, etc.
# -------------------------------------

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
        
        # --- INITIALIZE SPEECH ENGINE DIRECTLY ---
        # This copies the logic from tts_node.py
        self.language = os.environ.get("ASR_LANGUAGE", "English")
        self.get_logger().info(f"Initializing Speech Engine (Language: {self.language})...")
        
        try:
            if self.language == 'Chinese':
                self.tts_engine = speech.RealTimeTTS(log=self.get_logger())
            else:
                self.tts_engine = speech.RealTimeOpenAITTS(log=self.get_logger())
            self.get_logger().info("Speech Engine Ready!")
        except Exception as e:
            self.get_logger().error(f"Failed to init speech engine: {e}")
            self.tts_engine = None
        # -----------------------------------------

        # Initialize Hand Detector
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5
        )
        self.drawing = mp.solutions.drawing_utils
        
        # Publishers (Only Chassis needed now)
        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1) 
        
        # Camera Subscription
        self.camera_topic = '/ascamera/camera_publisher/rgb0/image'
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=2)
        
        # Dynamic import to handle message type
        from sensor_msgs.msg import Image
        self.create_subscription(Image, self.camera_topic, self.image_callback, 1)

        self.running = True
        self.is_backing_up = False 
        
        threading.Thread(target=self.image_proc, daemon=True).start()
        self.get_logger().info('Fist Detection Node Started. Show a FIST to say DANGER and move back!')

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

    def trigger_warning_and_move(self):
        if self.is_backing_up:
            return

        self.is_backing_up = True
        self.get_logger().warn("FIST DETECTED! Processing response...")

        # 1. DIRECT SPEECH GENERATION
        if self.tts_engine:
            self.get_logger().info("Generating Audio: 'Danger'")
            # Uses configuration from large_models.config
            try:
                self.tts_engine.tts("danger", model=tts_model, voice=voice_model)
            except Exception as e:
                 self.get_logger().error(f"Audio Generation Failed: {e}")

        # 2. Move Backwards
        twist = Twist()
        twist.linear.x = -0.2
        self.mecanum_pub.publish(twist)
        
        time.sleep(0.5) 
        
        # 3. Stop
        twist.linear.x = 0.0
        self.mecanum_pub.publish(twist)
        
        time.sleep(1.0) 
        self.is_backing_up = False

    def image_proc(self):
        while self.running:
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            image_flip = cv2.flip(image, 1)
            bgr_image = cv2.cvtColor(image_flip, cv2.COLOR_RGB2BGR)
            results = self.hand_detector.process(image_flip)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.drawing.draw_landmarks(
                        bgr_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    if self.is_fist(hand_landmarks.landmark, image_flip.shape):
                        threading.Thread(target=self.trigger_warning_and_move).start()
                        cv2.putText(bgr_image, "FIST: DANGER!", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow(self.name, bgr_image)
            key = cv2.waitKey(1)
            if key == 27:
                self.running = False

        self.mecanum_pub.publish(Twist()) 
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