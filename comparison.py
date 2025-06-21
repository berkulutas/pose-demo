import cv2
import mediapipe as mp
from ultralytics import YOLO
import time
import psutil
import os
import numpy as np

class PoseComparison:
    def __init__(self, video_path=None):
        # Initialize YOLOv8
        self.yolo_model = YOLO("yolov8n-pose.pt")
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.mediapipe_model = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Video capture
        self.cap = cv2.VideoCapture(video_path if video_path else 0)
        
        # Metrics storage
        self.yolo_times = []
        self.mediapipe_times = []
        self.yolo_detections = 0
        self.mediapipe_detections = 0
        self.frame_count = 0
        
        # Load overlay image
        try:
            self.overlay_img = cv2.imread("hat.png", cv2.IMREAD_UNCHANGED)
            if self.overlay_img is not None:
                self.hat_size = 80
                self.overlay_img = cv2.resize(self.overlay_img, (self.hat_size, self.hat_size))
            else:
                print("Warning: hat.png not found, skipping overlay")
        except:
            self.overlay_img = None
            print("Warning: Could not load hat.png")

    def overlay_transparent(self, background, overlay, x, y):
        """Overlay RGBA image over BGR background at (x, y)"""
        if overlay is None:
            return background
            
        bh, bw = background.shape[:2]
        oh, ow = overlay.shape[:2]

        if x < 0 or y < 0 or x + ow > bw or y + oh > bh:
            return background

        overlay_img = overlay[:, :, :3]
        mask = overlay[:, :, 3:] / 255.0
        roi = background[y:y+oh, x:x+ow]
        background[y:y+oh, x:x+ow] = (1 - mask) * roi + mask * overlay_img
        return background

    def process_yolo(self, frame):
        """Process frame with YOLOv8"""
        start_time = time.time()
        results = self.yolo_model(frame, verbose=False)
        inference_time = time.time() - start_time
        
        self.yolo_times.append(inference_time)
        
        # Get keypoints
        keypoints = results[0].keypoints
        num_persons = len(keypoints.xy) if keypoints and keypoints.xy is not None else 0
        
        if num_persons > 0:
            self.yolo_detections += 1
        
        # Create annotated frame
        annotated_frame = results[0].plot()
        
        # Add hat overlay if possible
        if keypoints and keypoints.xy is not None and self.overlay_img is not None:
            for person in keypoints.xy:
                if len(person) > 2:  # Check if eyes are detected
                    left_eye = person[1]
                    right_eye = person[2]
                    if not (torch.isnan(left_eye).any() or torch.isnan(right_eye).any()):
                        x = int(((left_eye[0] + right_eye[0]) / 2).item())
                        y = int(((left_eye[1] + right_eye[1]) / 2).item()) - 20
                        annotated_frame = self.overlay_transparent(
                            annotated_frame, self.overlay_img, 
                            x - self.hat_size // 2, y - self.hat_size // 2
                        )
        
        return annotated_frame, num_persons, inference_time

    def process_mediapipe(self, frame):
        """Process frame with MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        start_time = time.time()
        results = self.mediapipe_model.process(rgb_frame)
        inference_time = time.time() - start_time
        
        self.mediapipe_times.append(inference_time)
        
        pose_detected = results.pose_landmarks is not None
        num_persons = 1 if pose_detected else 0
        
        if pose_detected:
            self.mediapipe_detections += 1
        
        # Create annotated frame
        annotated_frame = frame.copy()
        if pose_detected:
            self.mp_draw.draw_landmarks(
                annotated_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            
            # Add hat overlay
            if self.overlay_img is not None:
                landmarks = results.pose_landmarks.landmark
                h, w, _ = frame.shape
                
                if len(landmarks) > 5:
                    left_eye = landmarks[2]
                    right_eye = landmarks[5]
                    
                    left_eye_x = int(left_eye.x * w)
                    left_eye_y = int(left_eye.y * h)
                    right_eye_x = int(right_eye.x * w)
                    right_eye_y = int(right_eye.y * h)
                    
                    x = int((left_eye_x + right_eye_x) / 2)
                    y = int((left_eye_y + right_eye_y) / 2) - 20
                    
                    annotated_frame = self.overlay_transparent(
                        annotated_frame, self.overlay_img,
                        x - self.hat_size // 2, y - self.hat_size // 2
                    )
        
        return annotated_frame, num_persons, inference_time

    def add_metrics_overlay(self, frame, title, fps, detection_rate, inference_time):
        """Add performance metrics to frame"""
        cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Detection: {detection_rate:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {inference_time*1000:.1f}ms", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

    def run_comparison(self):
        """Run the comparison"""
        print("Starting pose estimation comparison...")
        print("Press 'q' to quit, 's' to save current frame comparison")
        
        start_time = time.time()
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Process with both models
            yolo_frame, yolo_persons, yolo_time = self.process_yolo(frame.copy())
            mediapipe_frame, mp_persons, mp_time = self.process_mediapipe(frame.copy())
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            overall_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            yolo_fps = 1.0 / yolo_time if yolo_time > 0 else 0
            mp_fps = 1.0 / mp_time if mp_time > 0 else 0
            
            yolo_detection_rate = (self.yolo_detections / self.frame_count) * 100
            mp_detection_rate = (self.mediapipe_detections / self.frame_count) * 100
            
            # Add metrics overlay
            yolo_frame = self.add_metrics_overlay(yolo_frame, "YOLOv8 Pose", yolo_fps, yolo_detection_rate, yolo_time)
            mediapipe_frame = self.add_metrics_overlay(mediapipe_frame, "MediaPipe Pose", mp_fps, mp_detection_rate, mp_time)
            
            # Resize frames to same size for side-by-side display
            height = 480
            yolo_frame = cv2.resize(yolo_frame, (int(height * yolo_frame.shape[1] / yolo_frame.shape[0]), height))
            mediapipe_frame = cv2.resize(mediapipe_frame, (int(height * mediapipe_frame.shape[1] / mediapipe_frame.shape[0]), height))
            
            # Create side-by-side comparison
            comparison_frame = np.hstack([yolo_frame, mediapipe_frame])
            
            # Add overall metrics
            cv2.putText(comparison_frame, f"Overall FPS: {overall_fps:.1f} | Frame: {self.frame_count}", 
                       (10, comparison_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Pose Estimation Comparison: YOLOv8 vs MediaPipe", comparison_frame)
            
            # Print periodic updates
            if self.frame_count % 30 == 0:
                self.print_current_stats()
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"comparison_frame_{self.frame_count}.jpg", comparison_frame)
                print(f"Saved comparison frame {self.frame_count}")
        
        self.print_final_results()
        self.cleanup()

    def print_current_stats(self):
        """Print current performance statistics"""
        if len(self.yolo_times) > 0 and len(self.mediapipe_times) > 0:
            yolo_avg_fps = 1.0 / np.mean(self.yolo_times)
            mp_avg_fps = 1.0 / np.mean(self.mediapipe_times)
            
            print(f"\n--- Frame {self.frame_count} Stats ---")
            print(f"YOLOv8    - Avg FPS: {yolo_avg_fps:.1f}, Detection Rate: {(self.yolo_detections/self.frame_count)*100:.1f}%")
            print(f"MediaPipe - Avg FPS: {mp_avg_fps:.1f}, Detection Rate: {(self.mediapipe_detections/self.frame_count)*100:.1f}%")

    def print_final_results(self):
        """Print final comparison results"""
        print("\n" + "="*60)
        print("FINAL COMPARISON RESULTS")
        print("="*60)
        
        if len(self.yolo_times) > 0 and len(self.mediapipe_times) > 0:
            # Performance metrics
            yolo_avg_time = np.mean(self.yolo_times)
            yolo_avg_fps = 1.0 / yolo_avg_time
            yolo_std_time = np.std(self.yolo_times)
            
            mp_avg_time = np.mean(self.mediapipe_times)
            mp_avg_fps = 1.0 / mp_avg_time
            mp_std_time = np.std(self.mediapipe_times)
            
            # Detection rates
            yolo_detection_rate = (self.yolo_detections / self.frame_count) * 100
            mp_detection_rate = (self.mediapipe_detections / self.frame_count) * 100
            
            print(f"Total Frames Processed: {self.frame_count}")
            print(f"\nYOLOv8 Performance:")
            print(f"  Average FPS: {yolo_avg_fps:.2f}")
            print(f"  Average Inference Time: {yolo_avg_time*1000:.2f} ± {yolo_std_time*1000:.2f} ms")
            print(f"  Detection Rate: {yolo_detection_rate:.1f}%")
            print(f"  Total Detections: {self.yolo_detections}")
            
            print(f"\nMediaPipe Performance:")
            print(f"  Average FPS: {mp_avg_fps:.2f}")
            print(f"  Average Inference Time: {mp_avg_time*1000:.2f} ± {mp_std_time*1000:.2f} ms")
            print(f"  Detection Rate: {mp_detection_rate:.1f}%")
            print(f"  Total Detections: {self.mediapipe_detections}")
            
            print(f"\nComparison:")
            if yolo_avg_fps > mp_avg_fps:
                print(f"  🏆 YOLOv8 is {yolo_avg_fps/mp_avg_fps:.2f}x faster")
            else:
                print(f"  🏆 MediaPipe is {mp_avg_fps/yolo_avg_fps:.2f}x faster")
            
            if yolo_detection_rate > mp_detection_rate:
                print(f"  🎯 YOLOv8 has {yolo_detection_rate - mp_detection_rate:.1f}% higher detection rate")
            else:
                print(f"  🎯 MediaPipe has {mp_detection_rate - yolo_detection_rate:.1f}% higher detection rate")
        
        # System resources
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        cpu_percent = process.cpu_percent()
        
        print(f"\nSystem Resources:")
        print(f"  Memory Usage: {memory_usage:.1f} MB")
        print(f"  CPU Usage: {cpu_percent:.1f}%")
        
        print("="*60)

    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.mediapipe_model.close()

# Import torch for YOLOv8 (needed for tensor operations)
try:
    import torch
except ImportError:
    print("Warning: torch not found, YOLOv8 may not work properly")

if __name__ == "__main__":
    # Usage examples:
    # For webcam: comparison = PoseComparison()
    # For video file: comparison = PoseComparison("demo_video.mp4")
    
    comparison = PoseComparison("demo_video.mp4")  # Change to None for webcam
    comparison.run_comparison()