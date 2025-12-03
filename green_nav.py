#!/usr/bin/env python3
# encoding: utf-8
# Line following (green) + LiDAR obstacle avoidance

import os
import cv2
import math
import time
import rclpy
import queue
import threading
import numpy as np

import sdk.pid as pid
import sdk.common as common

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from app.common import Heart, ColorPicker
from cv_bridge import CvBridge

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger, SetBool

if os.environ['DEPTH_CAMERA_TYPE'] == 'aurora':
    from depth_camera_aurora.decoder import decode_image


lock = threading.RLock()


class LineFollower(object):
    def __init__(self, target_color, node):
        self.node = node
        self.target_lab, self.rois = target_color[0], None
        self.deflection_angle = None
        self.depth_camera_type = os.environ['DEPTH_CAMERA_TYPE']
        self.camera_type = os.environ['CAMERA_TYPE']

        # --- ROIs for different cameras ---
        if self.camera_type == 'Stereo':
            self.rois = (
                (0.70, 0.75, 0.4, 0.7, 0.7),
                (0.55, 0.60, 0.4, 0.7, 0.2),
                (0.40, 0.45, 0.4, 0.7, 0.1),
            )
        elif self.camera_type in ('Webcam_Stream', 'Realsense', 'Oak-D Pro W'):
            self.rois = (
                (0.8, 0.85, 0.1, 0.6, 0.7),
                (0.6, 0.65, 0.1, 0.55, 0.1),
                (0.4, 0.45, 0.1, 0.5, 0.1),
                (0.2, 0.25, 0.1, 0.45, 0.1),
            )
        elif self.depth_camera_type == 'ascamera':
            self.rois = (
                (0.9, 0.95, 0, 1, 0.7),
                (0.8, 0.85, 0, 1, 0.2),
                (0.7, 0.75, 0, 1, 0.1),
            )
        elif self.depth_camera_type == 'aurora':
            # original aurora ROIs; you can tweak if needed
            self.rois = (
                (0.81, 0.83, 0, 1, 0.7),
                (0.69, 0.71, 0, 1, 0.2),
                (0.57, 0.59, 0, 1, 0.1),
            )

        if isinstance(target_color[1], list):
            self.target_lab = target_color[0]
            self.lowerb = target_color[1]
            self.upperb = target_color[2]

    def get_area_max_contour(self, contours):
        area_max_contour, area_max = None, 0
        for c in contours:
            area = abs(cv2.contourArea(c))
            if area > area_max:
                area_max = area
                area_max_contour = c
        return area_max_contour, area_max

    def get_contour(self, binary, result_image, roi, x_bias=0, y_bias=0):
        image, contours, _ = common.get_objs_contours(binary)
        center_x = None
        if len(contours) > 0:
            area = max(contours, key=cv2.contourArea)
            theta = common.get_angle(area, image)
            if abs(theta) not in [90.0, 0.0] and abs(theta) < 50.0:
                (center_x, center_y), radius = cv2.minEnclosingCircle(area)
                center_x, center_y, radius = int(center_x), int(center_y), int(radius)

                cv2.circle(result_image, (center_x + x_bias, center_y + y_bias), radius, (0, 0, 255), 2)
                cv2.circle(result_image, (center_x + x_bias, center_y + y_bias), 5, (0, 0, 255), -1)
                cv2.drawContours(result_image, contours, -1, (255, 0, 0), 2)

                cv2.rectangle(
                    result_image,
                    (int(roi[2] * self.node.image_width) + x_bias, int(roi[0] * self.node.image_height) + y_bias),
                    (int(roi[3] * self.node.image_width) + x_bias, int(roi[1] * self.node.image_height) + y_bias),
                    (34, 139, 34),
                    2,
                )
        return result_image, center_x

    def __call__(self, image, result_image, threshold, color=None, use_color_picker=False):
        image_height, image_width = image.shape[:2]
        if self.target_lab is None:
            if color is None:
                return result_image, None
            else:
                self.target_lab = color
        if threshold < 0.1:
            threshold = 0.1
        threshold = min(threshold, 1.0)

        if use_color_picker:
            self.target_l_lab = [
                int(self.target_lab[0] - 20 * threshold),
                int(self.target_lab[1] - 20 * threshold),
                int(self.target_lab[2] - 20 * threshold),
            ]
            self.target_h_lab = [
                int(self.target_lab[0] + 20 * threshold),
                int(self.target_lab[1] + 20 * threshold),
                int(self.target_lab[2] + 20 * threshold),
            ]
            min_color = self.target_l_lab
            max_color = self.target_h_lab
            lowerb = tuple(self.target_l_lab)
            upperb = tuple(self.target_h_lab)
        else:
            min_color = [
                int(self.target_lab[0] - 50 * threshold * 2),
                int(self.target_lab[1] - 50 * threshold),
                int(self.target_lab[2] - 50 * threshold),
            ]
            max_color = [
                int(self.target_lab[0] + 50 * threshold * 2),
                int(self.target_lab[1] + 50 * threshold),
                int(self.target_lab[2] + 50 * threshold),
            ]
            if color is not None:
                min_color = color['min']
                max_color = color['max']

        min_color = np.clip(min_color, 0, 255)
        max_color = np.clip(max_color, 0, 255)
        color_dist = math.sqrt(
            (max_color[0] - min_color[0]) ** 2
            + (max_color[1] - min_color[1]) ** 2
            + (max_color[2] - min_color[2]) ** 2
        )
        threshold = common.set_range(color_dist, 0, 200, 0.1, 1.0)

        if hasattr(self, 'target_l_lab'):
            min_color = self.target_l_lab
            max_color = self.target_h_lab
        target_color = [self.target_lab, min_color, max_color]

        if hasattr(self, 'lowerb') and hasattr(self, 'upperb'):
            lowerb = tuple(self.lowerb)
            upperb = tuple(self.upperb)
        else:
            lowerb = tuple(color['min'])
            upperb = tuple(color['max'])

        for roi in self.rois:
            blob = image[
                int(roi[0] * image_height) : int(roi[1] * image_height),
                int(roi[2] * image_width) : int(roi[3] * image_width),
            ]
            img_lab = cv2.cvtColor(blob, cv2.COLOR_RGB2LAB)
            img_blur = cv2.GaussianBlur(img_lab, (3, 3), 3)
            mask = cv2.inRange(img_blur, lowerb, upperb)
            eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]
            blob_area_max_contour, blob_area_max = self.get_area_max_contour(contours)
            if blob_area_max > 50:
                result_image, center_x = self.get_contour(
                    dilated, result_image, roi, int(roi[2] * image_width), int(roi[0] * image_height)
                )
                if center_x is not None:
                    self.deflection_angle = common.get_deflection_angle(
                        image_width / 2,
                        center_x + int(roi[2] * image_width),
                        int(roi[1] * image_height),
                        int(roi[0] * image_height),
                    )
                    cv2.putText(
                        result_image,
                        'Angle: ' + str(int(self.deflection_angle / math.pi * 180)) + ' deg',
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        [0, 0, 255],
                    )
        return result_image, self.deflection_angle


class LineFollowingWithLidarNode(Node):
    def __init__(self, name):
        super().__init__(name, automatically_declare_parameters_from_overrides=True)

        self.name = name

        # --- line following state ---
        self.color = 'green'                # follow green line by default
        self.set_callback = False
        self.is_running = False
        self.color_picker = None
        self.follower = None
        self.pid = pid.PID(0.005, 0.001, 0.0)

        self.empty = 0
        self.count = 0
        self.stop = False                   # used by depth cam stop
        self.threshold = 0.5
        self.stop_threshold = 0.4

        self.lock = threading.RLock()
        self.image_height = None
        self.image_width = None
        self.bridge = CvBridge()
        self.use_color_picker = False       # we use named color "green" by default

        self.lab_data = common.get_yaml_data("/home/ubuntu/software/lab_tool/lab_config.yaml")
        self.image_queue = queue.Queue(2)
        self.camera_type = os.environ['CAMERA_TYPE']
        self.depth_camera_type = os.environ['DEPTH_CAMERA_TYPE']
        self.machine_type = os.environ['MACHINE_TYPE']
        self.wait_cv = threading.Condition(threading.Lock())
        self.result_image = None
        self.common = None

        # --- LiDAR obstacle avoidance state (from lidar_controller) ---
        self.obs_threshold = 0.6           # meters
        self.obs_scan_angle = math.radians(90)
        self.obs_speed = 0.2
        self.obs_last_act = 0
        self.obs_timestamp = 0.0
        self.lidar_type = os.environ.get('LIDAR_TYPE', '')
        self.avoid_twist = Twist()
        self.avoid_active_until = 0.0      # time until which avoidance overrides line following

        # --- ROS entities ---
        self.heart = Heart(self)
        self.mecanum_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)

        if self.depth_camera_type == 'ascamera':
            self.depth_subscriber = self.create_subscription(
                PointCloud2, '/oakd/depth/image_raw', self.depth_camera_callback, 10
            )
        elif self.depth_camera_type == 'aurora':
            self.depth_subscriber = self.create_subscription(
                Image, '/aurora/depth/image_raw', self.depth_camera_callback, 10
            )

        # LiDAR subscription with QoS like lidar_controller
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan_raw', self.lidar_callback, qos)

        # Services
        self.enter_server = self.create_service(Trigger, f'/{name}/enter', self.enter_srv_callback)
        self.exit_server = self.create_service(Trigger, f'/{name}/exit', self.exit_srv_callback)
        self.set_running_server = self.create_service(SetBool, f'/{name}/set_running', self.set_srv_callback)
        self.set_color_server = self.create_service(String, f'/{name}/set_color', self.set_color_srv_callback)
        self.set_threshold_server = self.create_service(String, f'/{name}/set_threshold', self.set_threshold_srv_callback)
        self.set_target_color_server = self.create_service(
            Trigger, f'/{name}/set_target_color', self.set_target_color_srv_callback
        )

    # ---------- Simple service callbacks ----------

    def set_threshold_srv_callback(self, request, response):
        data = eval(request.data)
        if 'threshold' in data:
            with self.lock:
                self.threshold = data['threshold']
        if 'stop_threshold' in data:
            with self.lock:
                self.stop_threshold = data['stop_threshold']
        response.success = True
        return response

    def set_color_srv_callback(self, request, response):
        with self.lock:
            self.color = request.data
            self.get_logger().info(f"Color set to: {self.color}")
        response.success = True
        return response

    def set_target_color_srv_callback(self, request, response):
        if self.color_picker is None:
            self.color_picker = ColorPicker()
        else:
            self.color_picker = None
        response.success = True
        return response

    # ---------- LiDAR-based obstacle avoidance (simplified from lidar_controller) ----------

    def lidar_callback(self, lidar_data: LaserScan):
        """
        Use LiDAR to generate an avoidance twist for a short time window.
        Logic adapted from running_mode == 1 in lidar_controller.py (non-Acker case).
        """
        twist = Twist()

        # Choose left/right slices according to lidar type
        if self.lidar_type != 'G4':
            # MAX_SCAN_ANGLE logic from lidar_controller
            if 'Pro' in os.environ.get('MACHINE_TYPE', ''):
                max_scan_angle_deg = 120
            else:
                max_scan_angle_deg = 240
            max_index = int(math.radians(max_scan_angle_deg / 2.0) / lidar_data.angle_increment)
            left_ranges = lidar_data.ranges[:max_index]
            right_ranges = lidar_data.ranges[::-1][:max_index]
        else:
            if 'Pro' in os.environ.get('MACHINE_TYPE', ''):
                max_scan_angle_deg = 120
            else:
                max_scan_angle_deg = 240
            min_index = int(
                math.radians((360 - max_scan_angle_deg) / 2.0) / lidar_data.angle_increment
            )
            max_index = min_index + int(
                math.radians(max_scan_angle_deg / 2.0) / lidar_data.angle_increment
            )
            left_ranges = lidar_data.ranges[::-1][min_index:max_index][::-1]
            right_ranges = lidar_data.ranges[min_index:max_index][::-1]

        with self.lock:
            # Only act when previous avoidance has finished
            if time.time() < self.obs_timestamp:
                return

            angle = self.obs_scan_angle / 2
            angle_index = int(angle / lidar_data.angle_increment + 0.5)
            left_range = np.array(left_ranges[:angle_index])
            right_range = np.array(right_ranges[:angle_index])

            # Only use non-zero finite values
            left_nonzero = left_range.nonzero()
            right_nonzero = right_range.nonzero()
            left_nonan = np.isfinite(left_range[left_nonzero])
            right_nonan = np.isfinite(right_range[right_nonzero])

            min_dist_left_ = left_range[left_nonzero][left_nonan]
            min_dist_right_ = right_range[right_nonzero][right_nonan]

            if len(min_dist_left_) <= 1 or len(min_dist_right_) <= 1:
                return

            min_dist_left = min_dist_left_.min()
            min_dist_right = min_dist_right_.min()

            # Same branching logic as lidar_controller (non-Acker, running_mode==1),
            # but instead of publishing here, we store twist & a time window.
            if min_dist_left <= self.obs_threshold and min_dist_right > self.obs_threshold:
                # obstacle on left -> turn right
                twist.linear.x = self.obs_speed / 6.0
                max_angle = math.radians(90)
                w = self.obs_speed * 6.0
                twist.angular.z = -w
                if self.obs_last_act != 0 and self.obs_last_act != 1:
                    twist.angular.z = w
                self.obs_last_act = 1
                duration = max_angle / w / 2.0

            elif min_dist_left <= self.obs_threshold and min_dist_right <= self.obs_threshold:
                # obstacles on both sides -> turn around
                twist.linear.x = self.obs_speed / 6.0
                w = self.obs_speed * 6.0
                twist.angular.z = w
                self.obs_last_act = 3
                duration = math.radians(180) / w / 2.0

            elif min_dist_left > self.obs_threshold and min_dist_right <= self.obs_threshold:
                # obstacle on right -> turn left
                twist.linear.x = self.obs_speed / 6.0
                max_angle = math.radians(90)
                w = self.obs_speed * 6.0
                twist.angular.z = w
                if self.obs_last_act != 0 and self.obs_last_act != 2:
                    twist.angular.z = -w
                self.obs_last_act = 2
                duration = max_angle / w / 2.0

            else:
                # No obstacle nearby -> no special avoidance; let line-following run
                self.obs_last_act = 0
                return

            # Set avoidance twist + time window
            self.avoid_twist = twist
            self.avoid_active_until = time.time() + duration
            self.obs_timestamp = self.avoid_active_until

    # ---------- Depth camera obstacle stop (existing behaviour) ----------

    def depth_camera_callback(self, image):
        if self.depth_camera_type == 'ascamera':
            img = self.bridge.imgmsg_to_cv2(image, '32FC1')
            img = common.convert_ascamera_depth_16bit(image, img)
        elif self.depth_camera_type == 'aurora':
            img = self.bridge.imgmsg_to_cv2(image, 'mono16')
            img = decode_image(img)

        with self.lock:
            if self.image_height is None or self.image_width is None:
                return
            exterior_corners = [
                (360, 0),
                (518, 0),
                (self.image_width, 240),
                (self.image_width, 374),
                (0, self.image_height),
            ]
            mask_image = np.zeros((self.image_height, self.image_width, 1), dtype="uint8")
            cv2.fillPoly(mask_image, [np.array(exterior_corners)], (255,))
            avg_depth_value = np.average(np.multiply(img, mask_image) / 255)

            if avg_depth_value < 0.6:
                self.stop = True
                twist = Twist()
                self.mecanum_pub.publish(twist)
            else:
                self.stop = False

    # ---------- Enter / Exit / Running services ----------

    def enter_srv_callback(self, request, response):
        response.success = True
        self.set_callback = False
        self.is_running = False
        self.count = 0
        self.color_picker = None
        self.follower = None
        self.image_queue = queue.Queue(2)
        self.stop = False
        self.threshold = 0.5
        self.stop_threshold = 0.4
        self.common = common.CommonNode(self)
        self.color = 'green'

        # reset lidar avoidance state
        self.obs_last_act = 0
        self.obs_timestamp = 0.0
        self.avoid_active_until = 0.0

        return response

    def exit_srv_callback(self, request, response):
        response.success = True
        self.set_callback = False
        self.is_running = False
        self.empty = 0
        self.count = 0
        self.color_picker = None
        self.follower = None
        self.heart.lose()
        self.stop = False
        self.threshold = 0.5
        self.stop_threshold = 0.4
        self.common = None
        # stop robot
        self.mecanum_pub.publish(Twist())
        return response

    def set_srv_callback(self, request, response):
        self.is_running = request.data
        self.get_logger().info("is_running: {}".format(self.is_running))
        response.success = True
        return response

    # ---------- Image queue helper ----------

    def image_queue_put(self, message):
        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(message)

    def image_process(self):
        while True:
            data = self.image_queue.get()
            if data is None:
                break
            self.image_callback(data, from_queue=True)

    # ---------- Main image callback: line following + lidar override ----------

    def image_callback(self, ros_image, from_queue=False):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        self.image_height, self.image_width = rgb_image.shape[:2]
        result_image = np.copy(rgb_image)

        with self.lock:
            twist = Twist()

            if self.use_color_picker:
                # (Picker path kept for completeness, but default is named color "green")
                if self.color_picker is not None:
                    try:
                        target_color, result_image = self.color_picker(rgb_image, result_image)
                        if target_color is not None:
                            self.color_picker = None
                            self.follower = LineFollower(target_color, self)
                            self.get_logger().info("target color: {}".format(target_color))
                    except Exception as e:
                        self.get_logger().error(str(e))
                else:
                    twist.linear.x = 0.15
                    if self.follower is not None:
                        try:
                            result_image, deflection_angle = self.follower(
                                rgb_image, result_image, self.threshold
                            )
                            if deflection_angle is not None and self.is_running and not self.stop:
                                self.pid.update(deflection_angle)
                                if 'Acker' in self.machine_type:
                                    steering_angle = common.set_range(
                                        -self.pid.output, -math.radians(40), math.radians(40)
                                    )
                                    if steering_angle != 0:
                                        R = 0.145 / math.tan(steering_angle)
                                        twist.angular.z = twist.linear.x / R
                                else:
                                    twist.angular.z = common.set_range(-self.pid.output, -1.0, 1.0)
                            elif self.stop:
                                twist = Twist()
                            else:
                                self.pid.clear()
                        except Exception as e:
                            self.get_logger().error(str(e))
            else:
                # Named-color path: follow "green"
                if self.color in common.range_rgb:
                    twist.linear.x = 0.15
                    self.follower = LineFollower([None, common.range_rgb[self.color]], self)
                    result_image, deflection_angle = self.follower(
                        rgb_image,
                        result_image,
                        self.threshold,
                        self.lab_data['lab'][self.camera_type][self.color],
                        False,
                    )
                    if deflection_angle is not None and self.is_running and not self.stop:
                        self.pid.update(deflection_angle)
                        if 'Acker' in self.machine_type:
                            steering_angle = common.set_range(
                                -self.pid.output, -math.radians(40), math.radians(40)
                            )
                            if steering_angle != 0:
                                R = 0.145 / math.tan(steering_angle)
                                twist.angular.z = twist.linear.x / R
                        else:
                            twist.angular.z = common.set_range(-self.pid.output, -1.0, 1.0)
                    elif self.stop:
                        twist = Twist()
                    else:
                        self.pid.clear()

            # ---- LiDAR override: if avoidance window is active, replace twist ----
            now = time.time()
            if now < self.avoid_active_until:
                twist = self.avoid_twist

            # Publish final command
            if self.is_running:
                self.mecanum_pub.publish(twist)

        if not from_queue:
            self.image_queue_put(ros_image)

        # Debug window (unchanged from your original)
        if 'display' in os.environ and os.environ['display'] == '1':
            if os.environ['DEPTH_CAMERA_TYPE'] == 'ascamera':
                if not self.set_callback:
                    cv2.namedWindow("result", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.moveWindow("result", 480, 300)
                    cv2.resizeWindow("result", 800, 480)
                    self.common.add_mouse_callback(
                        'result',
                        self.image_width,
                        self.image_height,
                        self.set_target_color_server,
                    )
                    cv2.setMouseCallback("result", self.common.mouse_callback)
                    self.set_callback = True
            else:
                if not self.set_callback:
                    cv2.namedWindow("result", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.moveWindow("result", 480, 300)
                    cv2.resizeWindow("result", 800, 480)
                    self.set_callback = True
                cv2.imshow("result", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

    def destroy_node(self):
        self.heart.lose()
        super().destroy_node()


def main(args=None):
        rclpy.init(args=args)
        node = LineFollowingWithLidarNode("line_following_with_lidar")
        if 'display' in os.environ and os.environ['display'] == '1':
            image_process_thread = threading.Thread(target=node.image_process, daemon=True)
            image_process_thread.start()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
