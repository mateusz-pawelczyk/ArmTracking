import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import csv

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(min_detection_confidence=0.55, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55)
mp_drawing = mp.solutions.drawing_utils

def start_recording():
    global csvfile, csv_writer, recording
    csvfile = open('recorded_data.csv', 'w', newline='')
    csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Landmark', 'X', 'Y', 'Z', 'Depth'])
    recording = True
    print("Recording started.")

def stop_recording():
    global csvfile, recording
    csvfile.close()
    recording = False
    print("Recording stopped.")

def point_in_screen(point_x,point_y):
    return 0<= point_x <=1 and 0<= point_y <= 1

def z_relative_to_absolute(z_relational_point_absolute, z_relational_point_relative, z_calculate_relative):
    return (abs(z_relational_point_absolute/z_relational_point_relative) * z_calculate_relative)

recording = False

CALIBRATION_FACTOR_HAND = None

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(color_image_rgb)
        pose_results = pose.process(color_image_rgb)

        # Visualize and record arm landmarks with depth
        if pose_results.pose_landmarks:
            connections = [(mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                           (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)]
            for connection in connections:           
                start_point = pose_results.pose_landmarks.landmark[connection[0]]
                end_point = pose_results.pose_landmarks.landmark[connection[1]]
                x1, y1 = int(start_point.x * color_image.shape[1]), int(start_point.y * color_image.shape[0])
                x2, y2 = int(end_point.x * color_image.shape[1]), int(end_point.y * color_image.shape[0])
                

                cv2.line(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                if point_in_screen(x1,y1) and point_in_screen(x2,y2):
                    depth1 = depth_frame.get_distance(x1, y1)
                    depth2 = depth_frame.get_distance(x2, y2)
                    if not CALIBRATION_POINT_HAND and connection[0] == mp_pose.PoseLandmark.RIGHT_WRIST:
                        CALIBRATION_POINT_HAND = abs(depth1/ start_point.z)
                    elif not CALIBRATION_POINT_HAND and connection[1] == mp_pose.PoseLandmark.RIGHT_WRIST:
                        CALIBRATION_POINT_HAND = abs(depth2/ end_point.z)
                    cv2.putText(color_image, f'{depth1:.2f}m', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(color_image, f'{depth2:.2f}m', (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if recording:
                    csv_writer.writerow(['Pose', connection[0], start_point.x, start_point.y, start_point.z, depth1])
                    csv_writer.writerow(['Pose', connection[1], end_point.x, end_point.y, end_point.z, depth2])


        # Visualize hands without displaying depth information
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for id, landmark in enumerate(hand_landmarks.landmark):
                    if CALIBRATION_FACTOR_HAND:
                        absolute_depth = abs(landmark.z) *CALIBRATION_FACTOR_HAND
                        if id == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                            cv2.putText(color_image, f'{absolute_depth:.2f}m', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    x = int(landmark.x * color_image.shape[1])
                    y = int(landmark.y * color_image.shape[0])
                    
                    color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

        
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if not recording:
                start_recording()
            else:
                stop_recording()
        elif key == ord('q'):
            if recording:
                stop_recording()
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    if recording:
        csvfile.close()
