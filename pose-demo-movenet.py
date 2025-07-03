
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time

# Load MoveNet model from TensorFlow Hub
print("Loading MoveNet model...")
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']
print("Model loaded!")

# Pose connections for drawing skeleton
CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

def preprocess_image(image):
    """Resize and preprocess image for MoveNet"""
    # Resize to 192x192 (MoveNet input size)
    input_image = cv2.resize(image, (192, 192))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Convert to tensor and add batch dimension
    input_tensor = tf.cast(input_image, dtype=tf.int32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)

    return input_tensor

def draw_pose(image, keypoints, confidence_threshold=0.3):
    """Draw pose keypoints and skeleton on image"""
    h, w = image.shape[:2]

    # Draw skeleton connections
    for connection in CONNECTIONS:
        kp1_idx, kp2_idx = connection

        if (keypoints[kp1_idx][2] > confidence_threshold and 
            keypoints[kp2_idx][2] > confidence_threshold):

            # Get pixel coordinates
            y1, x1 = keypoints[kp1_idx][:2]
            y2, x2 = keypoints[kp2_idx][:2]

            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)

            # Draw line
            cv2.line(image, (px1, py1), (px2, py2), (0, 255, 0), 2)

    # Draw keypoints
    for i, (y, x, conf) in enumerate(keypoints):
        if conf > confidence_threshold:
            px, py = int(x * w), int(y * h)
            cv2.circle(image, (px, py), 4, (0, 0, 255), -1)

    return image

# Open camera
cap = cv2.VideoCapture('/home/pir/dev/pose-demo/demo_video.mp4')

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    input_tensor = preprocess_image(frame)

    # Inference with timing
    start = time.time()
    outputs = movenet(input_tensor)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]
    end = time.time()

    inference_time = end - start
    print(f"Inference time: {inference_time:.4f} seconds")


    # Draw pose on frame
    pose_frame = draw_pose(frame.copy(), keypoints)

    # Show result
    cv2.imshow('Simple PoseNet', pose_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
