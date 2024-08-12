import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import csv
import pandas as pd

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.55, min_tracking_confidence=0.5)
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

CALIBRATION_FACTOR_SHOULDER = None
CALIBRATION_FACTOR_WRIST = None
CALIBRATION_FACTOR_FINGER = 0
CALIBRATION_FACTOR_FINGER_INDEX = 0
CALIBRATION_FACTOR_FINGER_SUM = 0

ROW_CSV = []

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
            connections = [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW]
            for landmark in pose_results.pose_landmarks.landmark:           
                x, y = int(landmark.x * color_image.shape[1]), int(landmark.y * color_image.shape[0])
                
                if point_in_screen(landmark.x, landmark.y):
                    depth1 = depth_frame.get_distance(x, y)

                    cv2.putText(color_image, f'{depth1:.2f}m', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    ROW_CSV.append(x)

                if recording:
                    csv_writer.writerow(['Pose', connection[0], start_point.x, start_point.y, start_point.z, depth1])
                    csv_writer.writerow(['Pose', connection[1], end_point.x, end_point.y, end_point.z, depth2])


        # Visualize hands without displaying depth information
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Korrekter Aufruf von mp_drawing.draw_landmarks
                mp_drawing.draw_landmarks(
                    color_image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                z_fingertip_rel = 0
                z_wrist_rel = 0
                z_fingertip_abs = 0
                z_wrist_abs = 0

                z_wrist_rel_temp = 0
                z_fingertip_rel_temp = 0
                
                for id, landmark in enumerate(hand_landmarks.landmark):
                    x = int(landmark.x * color_image.shape[1])
                    y = int(landmark.y * color_image.shape[0])
                    #print(f"{CALIBRATION_FACTOR_SHOULDER=}")
                    #print(f"{landmark.z=}")
                    
                    if id == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                        depth= depth_frame.get_distance(x, y)
                        z_fingertip_rel_temp = landmark.z
                        z_fingertip_rel = z_fingertip_rel_temp
                        z_fingertip_abs = depth
                    if id == mp_hands.HandLandmark.WRIST:
                        depth= depth_frame.get_distance(x, y)
                        z_wrist_rel_temp = landmark.z
                        z_wrist_rel = z_wrist_rel_temp
                        z_wrist_abs = depth
                    if CALIBRATION_FACTOR_FINGER:
                        depth= depth_frame.get_distance(x, y)
                        absolute_depth = CALIBRATION_FACTOR_FINGER * (z_fingertip_rel - z_wrist_rel) + z_wrist_abs
                        if id == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                            cv2.putText(color_image, f'{absolute_depth:.2f}m', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                if z_fingertip_rel and z_wrist_rel and z_fingertip_abs and z_wrist_abs:
                    CALIBRATION_FACTOR_FINGER_SUM += abs(z_fingertip_abs - z_wrist_abs) / abs(z_fingertip_rel - z_wrist_rel)
                    CALIBRATION_FACTOR_FINGER_INDEX += 1
                    CALIBRATION_FACTOR_FINGER = CALIBRATION_FACTOR_FINGER_SUM / CALIBRATION_FACTOR_FINGER_INDEX
                    print(CALIBRATION_FACTOR_FINGER)

        
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if not recording:
                start_recording()
            else:
                stop_recording()
        elif key == ord('q'):
            df = pd.DataFrame(abs_and_rel, columns=['Depth', 'Landmark.z', 'Ratio'])
            df.to_csv('abs_to_rel.csv', index=False)
            if recording:
                stop_recording()
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    if recording:
        csvfile.close()
