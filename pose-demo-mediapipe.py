import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # 0=lite, 1=full, 2=heavy
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open video file
video_path = "demo_video.mp4"
cap = cv2.VideoCapture(video_path)

# ######## Uncomment the following line to use webcam and comment the video path line ########
# cap = cv2.VideoCapture(0)

overlay_img = cv2.imread("hat.png", cv2.IMREAD_UNCHANGED)  # Load overlay image 4 channels (RGBA)
print("Overlay image shape:", overlay_img.shape)

# Ensure overlay image is in RGBA format
if overlay_img.shape[2] != 4:
    raise ValueError("Overlay image must have 4 channels (RGBA)")

# Resize the hat to a smaller, fixed size (e.g., 60x60 px)
hat_size = 80
overlay_img = cv2.resize(overlay_img, (hat_size, hat_size), interpolation=cv2.INTER_AREA)


def overlay_transparent(background, overlay, x, y):
    """Overlay RGBA image over BGR background at (x, y)"""
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    # Clip overlay if it goes outside background
    if x < 0 or y < 0 or x + ow > bw or y + oh > bh:
        return background

    # Split overlay channels
    overlay_img = overlay[:, :, :3]
    mask = overlay[:, :, 3:] / 255.0  # alpha channel

    # Overlay region of interest
    roi = background[y:y+oh, x:x+ow]
    background[y:y+oh, x:x+ow] = (1 - mask) * roi + mask * overlay_img
    return background


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Inference
    start = time.time()
    results = pose.process(rgb_frame)
    end = time.time()
    print(f"Inference time: {(end - start):.4f} seconds")
    
    # Check if pose is detected
    pose_detected = results.pose_landmarks is not None
    
    # Overlay detection status
    if pose_detected:
        msg = "Pose Detected (1 person)"
        color = (0, 255, 0)
    else:
        msg = "No Pose Detected"
        color = (0, 0, 255)

    # Draw pose landmarks on frame
    annotated_frame = frame.copy()
    if pose_detected:
        # Draw pose landmarks and connections
        mp_draw.draw_landmarks(
            annotated_frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
        )

    # Put text on top-left
    cv2.putText(annotated_frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Draw hat if head keypoints are available
    if pose_detected and results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape
        
        # MediaPipe pose landmark indices:
        # 1: LEFT_EYE_INNER, 2: LEFT_EYE, 4: RIGHT_EYE_INNER, 5: RIGHT_EYE
        # We'll use LEFT_EYE (2) and RIGHT_EYE (5)
        if len(landmarks) > 5:
            left_eye = landmarks[2]
            right_eye = landmarks[5]
            
            # Convert normalized coordinates to pixel coordinates
            left_eye_x = int(left_eye.x * w)
            left_eye_y = int(left_eye.y * h)
            right_eye_x = int(right_eye.x * w)
            right_eye_y = int(right_eye.y * h)
            
            # Calculate center point between eyes
            x = int((left_eye_x + right_eye_x) / 2)
            y = int((left_eye_y + right_eye_y) / 2) - 20  # shift up a little
            
            # Overlay hat
            annotated_frame = overlay_transparent(
                annotated_frame, 
                overlay_img, 
                x - hat_size // 2, 
                y - hat_size // 2
            )

    # Show the result
    cv2.imshow("MediaPipe Pose Estimation", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()