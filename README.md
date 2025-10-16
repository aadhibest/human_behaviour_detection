# Human Behaviour Detection System

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

A sophisticated computer vision system that detects and analyzes human behavior through real-time video monitoring, focusing on face detection, eye tracking, and person/phone detection.

## Features

- **Face Detection**: Uses OpenCV's DNN module with Caffe models for accurate face detection
- **Eye Tracking**: Monitors eye movements and gaze direction
- **Person and Phone Detection**: Uses YOLO v3 to detect people and phones in the frame
- **Real-time Processing**: Processes video feed in real-time with efficient algorithms
- **Warning System**: Implements a warning system for behavior monitoring
- **Multiple Detection Models**: Integrates different pre-trained models for various detection tasks

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aadhibest/human_behaviour_detection.git
cd human_behaviour_detection
```

2. Install required dependencies:
```bash
pip install opencv-python
pip install tensorflow
pip install numpy
```

3. Download required model files:
   - The models folder should contain:
     - `res10_300x300_ssd_iter_140000.caffemodel` (Face detection)
     - `deploy.prototxt` (Face detection configuration)
     - `yolov3.weights` (YOLO model)
     - Other supporting model files

## Usage

1. Run the main program:
```bash
python final_pgm.py
```

2. The system will:
   - Access your webcam feed
   - Detect faces in real-time
   - Track eye movements
   - Detect people and phones
   - Save the output video as 'Final result.avi'


## Author

- [Aadhithya Krishnakumar](https://github.com/aadhibest/)
- [Navneeth Krishna](https://github.com/Navneeth-Krishna/)


---

⚠️ Note: This system requires a webcam and sufficient computational resources to run efficiently. Some models may need to be downloaded separately due to size constraints.