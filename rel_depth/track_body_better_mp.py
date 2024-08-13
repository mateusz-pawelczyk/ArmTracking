import pyrealsense2 as rs
import cv2
import mediapipe as mp
import numpy as np
import csv
import time

# Initialize Mediapipe and RealSense pipeline
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
pipeline = rs.pipeline()
config = rs.config()

# Configure the RealSense camera to stream RGB data
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Define a CSV file to store the joint coordinates
output_file = "../data/joint_coordinates.csv"
recording = False
header_written = False

try:
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        while True:
            # Capture frames from the camera
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert the frames to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Process the frame with Mediapipe Pose
            results = pose.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

            # Draw the pose annotation on the image
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # If recording, save the coordinates
                if recording:
                    frame_data = [time.time()]  # Record the timestamp
                    for landmark in results.pose_landmarks.landmark:
                        frame_data.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                    
                    if not header_written:
                        # Write the header with joint names
                        header = ["timestamp"]
                        for i in range(len(results.pose_landmarks.landmark)):
                            header.extend([f"x_{i}", f"y_{i}", f"z_{i}", f"visibility_{i}"])
                        writer.writerow(header)
                        header_written = True
                    
                    writer.writerow(frame_data)

            # Display the resulting frame
            cv2.imshow('RealSense', color_image)

            # Check for key press
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                if not recording:
                    print("Recording started...")
                else:
                    print("Recording stopped.")
                recording = not recording

            if key == 27:  # Press ESC to exit
                break

finally:
    # Stop the pipeline and close all windows
    pipeline.stop()
    cv2.destroyAllWindows()
