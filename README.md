# Pose Estimation Demo

This project demonstrates pose estimation using a deep learning model.

## Setup Instructions

Follow the steps below to set up and run the demo:

### 1. Create a Python Virtual Environment

```bash
python3 -m venv .venv
source venv/bin/activate
```


### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the Demo
```bash
python pose-demo.py
```

## Model Selection and Comparison
> Note:
I was unable to run the original PoseNet model on my Mac (M3 chip) due to TensorFlow compatibility issues. After researching alternatives, I found that MediaPipe Pose is a modern, actively maintained, and very fast pose estimation model. 

### Models Compared

- YOLOv8 Pose (Ultralytics):
    - Supports multi-person detection
- MediaPipe Pose (Google):
    - Single-person detection
    - Extremely fast and lightweight
    - Designed for real-time applications and mobile/edge devices



## FINAL COMPARISON RESULTS

```Total Frames Processed: 7662

YOLOv8 Performance:
  Average FPS: 36.87
  Average Inference Time: 27.13 ± 8.34 ms
  Detection Rate: 100.0%
  Total Detections: 7662

MediaPipe Performance:
  Average FPS: 75.63
  Average Inference Time: 13.22 ± 0.91 ms
  Detection Rate: 100.0%
  Total Detections: 7659

Comparison:
  🏆 MediaPipe is 2.05x faster
  🎯 YOLOv8 has 0.0% higher detection rate```