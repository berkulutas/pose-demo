import cv2
from ultralytics import YOLO

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt") 
# Open video file
video_path = "demo_video.mp4"
cap = cv2.VideoCapture(video_path)

# ######## Uncomment the following line to use webcam and comment the video path line ########
# # Open webcam
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
    if x + ow > bw or y + oh > bh:
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

    # Inference
    start = cv2.getTickCount()
    results = model(frame)
    end = cv2.getTickCount()
    print(f"Inference time: {(end - start) / cv2.getTickFrequency():.4f} seconds")
    
    # Get keypoints
    keypoints = results[0].keypoints
    num_persons = len(keypoints) if keypoints else 0

    # Overlay detection status
    if num_persons > 0:
        msg = f"Pose Detected ({num_persons} person)"
        color = (0, 255, 0)
    else:
        msg = "No Pose Detected"
        color = (0, 0, 255)
        continue

    # Annotate frame with keypoints
    annotated_frame = results[0].plot() # Get annotated frame with keypoints

    # Put text on top-left
    cv2.putText(annotated_frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Draw hat if head  available
    if keypoints:
        for person in keypoints.xy:
            if person.shape[0] > 2:  # Check both eyes
                left_eye = person[1]
                right_eye = person[2]
                x = int(((left_eye[0] + right_eye[0]) / 2).item())
                y = int(((left_eye[1] + right_eye[1]) / 2).item()) - 20  # shift up little
                annotated_frame = overlay_transparent(annotated_frame, overlay_img, x - hat_size // 2, y - hat_size // 2)
                # # debug: place hat top left
                # annotated_frame = overlay_transparent(annotated_frame, overlay_img, 50, 50)

    # Show the result
    cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
