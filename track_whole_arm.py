import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Enable both depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(min_detection_confidence=0.55, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55)
mp_drawing = mp.solutions.drawing_utils

def norm_vector_to_other_vector(v_to_norm_on, v_to_norm):
    print(f"v_to_norm_on: {v_to_norm_on}")
    print(f"v_to_norm: {v_to_norm}")
    print(f"v_to_norm_on / np.linalg.norm(v_to_norm_on): {v_to_norm_on / np.linalg.norm(v_to_norm_on)}")
    print((v_to_norm[0],v_to_norm[1], (v_to_norm_on / np.linalg.norm(v_to_norm_on))@v_to_norm))
    return (v_to_norm[0],v_to_norm[1], (v_to_norm_on / np.linalg.norm(v_to_norm_on))@v_to_norm)

VECTOR_FOR_CALIBRATION = ()


try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        y_mid_of_frame = int(0.5 * color_image.shape[0])
        x_mid_of_frame = int(0.5 * color_image.shape[1])
        depth = depth_frame.get_distance(x_mid_of_frame, y_mid_of_frame)
        VECTOR_FOR_CALIBRATION = (x_mid_of_frame, y_mid_of_frame, depth)
        if depth != 0:
            break

    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert the BGR image to RGB before processing with MediaPipe
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(color_image_rgb)
        pose_results = pose.process(color_image_rgb)

         # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for landmark in hand_landmarks.landmark:
                    if 0<= landmark.x <= 1 and 0<= landmark.y <= 1:
                        x = int(landmark.x * color_image.shape[1])
                        y = int(landmark.y * color_image.shape[0])
                        depth = depth_frame.get_distance(x, y)
                        #cv2.putText(color_image, f'{depth:.2f}m', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw arm landmarks and connectors from pose detection
        if pose_results.pose_landmarks:
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            arm_landmarks = [right_shoulder, right_elbow, right_wrist]
            arm_points = []

            for landmark in arm_landmarks:
                if landmark.x >= 0 and landmark.x <= 1 and landmark.y >= 0 and landmark.y <= 1:
                    x = int(landmark.x * color_image.shape[1])
                    y = int(landmark.y * color_image.shape[0])
                    depth = depth_frame.get_distance(x, y)
                    _,_, actual_depth = norm_vector_to_other_vector(VECTOR_FOR_CALIBRATION,(x,y,depth))
                    cv2.putText(color_image, f'{actual_depth:.2f}m', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.circle(color_image, (x, y), 5, (255, 0, 0), cv2.FILLED)
                    arm_points.append((x, y))

            # Draw connectors between shoulder, elbow, and wrist
            if len(arm_points) == 3:
                cv2.line(color_image, arm_points[0], arm_points[1], (255, 0, 0), 2)
                cv2.line(color_image, arm_points[1], arm_points[2], (255, 0, 0), 2)

        # Display the color image with landmarks and depth values
        cv2.imshow('RealSense', color_image)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
