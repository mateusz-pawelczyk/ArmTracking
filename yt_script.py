import pyrealsense2 as rs
import cv2
import mediapipe as mp
import numpy as np
import csv
import time

joint_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer", "left_ear", "right_ear",
    "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", 
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index"
]

# Initialize Mediapipe Pose and Hands models
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands()

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

output_file = "joint_coordinates.csv"
recording = False
header_written = False

def show_depth_value(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        depth = depth_frame.get_distance(x, y)
        print(f"Depth at ({x}, {y}): {depth:.3f} meters")

try:
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(color_image_rgb)
            hand_results = hands.process(color_image_rgb)

            if pose_results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    color_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('RealSense', color_image)
            cv2.imshow('RealSenseDepth', depth_colormap)
            cv2.setMouseCallback('RealSenseDepth', show_depth_value)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                recording = not recording
                print("Recording " + ("started..." if recording else "stopped."))
            if key == 27:
                break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
