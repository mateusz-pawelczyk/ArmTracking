import pyrealsense2 as rs
import mediapipe as mp
import numpy as np
import csv
import cv2

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Create a depth align object to align depth frames with color frames
align = rs.align(rs.stream.color)

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Output file for saving coordinates
output_file = 'hand_tracking_data.csv'
csvfile = None
csv_writer = None
recording = False
frame_count = 0

recording_num = 0

def start_recording():
    global csvfile, csv_writer, recording, frame_count, recording_num
    file = f'hand_tracking_data_{recording_num}.csv'
    csvfile = open(file, mode='w', newline='')
    csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Frame', 'X', 'Y', 'Z'])  # Write header
    recording = True
    frame_count = 0
    recording_num += 1
    print("Recording started. Press 's' to stop recording.")

def stop_recording():
    global csvfile, recording
    if csvfile:
        csvfile.close()
    recording = False
    print(f"Recording stopped. Hand tracking data saved to {output_file}")

# Start recording initially
#start_recording()

try:
    while True:
        # Wait for the next set of frames from the camera
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert depth data to a numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert color image to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Convert color image to RGB for MediaPipe
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Process hand detection with MediaPipe
        results = hands.process(rgb_image)

        # Draw hand landmarks on the color image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Convert normalized landmarks to pixel coordinates
                height, width, _ = color_image.shape
                landmarks_px = []
                for landmark in hand_landmarks.landmark:
                    x_px, y_px = (landmark.x, landmark.y)
                    landmarks_px.append((x_px, y_px))

                # Calculate depth at specific landmarks (e.g., index finger tip)
                index_finger_tip = landmarks_px[8]  # index finger tip landmark

                if index_finger_tip[0] > 1 or index_finger_tip[0] < 0 or index_finger_tip[1] > 1 or index_finger_tip[1] < 0:
                    depth_value = 0
                else:
                    depth_value = depth_frame.get_distance(int(index_finger_tip[0] * width), int(index_finger_tip[1] * height))

                # Save coordinates and depth value to CSV if recording is enabled
                if recording:
                    csv_writer.writerow([frame_count, index_finger_tip[0], index_finger_tip[1], depth_value])
                    frame_count += 1

                # Print coordinates and depth value
                print(f"Coordinates (x, y): ({index_finger_tip[0]}, {index_finger_tip[1]}), Depth Value: {depth_value:.2f} meters")

                # Draw landmarks on color image
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display depth and color frames
        cv2.imshow('MediaPipe Hand Tracking', color_image)

        # Check for key press to start/stop recording ('s' key)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if recording:
                stop_recording()
            else:
                start_recording()

        # Exit loop on 'q' press
        elif key == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()

    # Close CSV file if still open
    if csvfile:
        csvfile.close()
