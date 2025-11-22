# Module 2: Face Detection

## Overview

In this module, you'll dive into one of the most popular applications of computer vision: face detection. You'll learn how to use pre-trained models to detect human faces in images and video streams in real-time. This module introduces you to the concept of using pre-trained models and cascade classifiers.

## Learning Objectives

By the end of this module, you will:

- Understand the principles of Haar cascade classifiers
- Use OpenCV's pre-trained face detection models
- Implement real-time face detection from webcam feeds
- Learn about different face detection algorithms and their trade-offs
- Explore face landmark detection
- Handle multiple faces in a single frame

## Theory Background

### Haar Cascade Classifiers

Haar cascades are machine learning-based approaches where a cascade function is trained from positive and negative images. The algorithm uses Haar-like features to detect objects:

- **Haar-like Features**: Rectangular features that capture differences in intensity
- **Integral Images**: Efficient computation of rectangular region sums
- **Cascade Structure**: Multiple stages of classifiers for efficiency
- **AdaBoost**: Machine learning algorithm used for training

### Face Detection Pipeline

1. **Preprocessing**: Convert image to grayscale, normalize
2. **Multi-scale Detection**: Search at different scales
3. **Feature Detection**: Apply Haar-like features
4. **Classification**: Use trained cascade to classify regions
5. **Non-Maximum Suppression**: Remove overlapping detections

### Modern Approaches

- **DNN-based Detection**: Deep learning models (SSD, MTCNN)
- **Face Landmarks**: 68-point facial landmark detection
- **Face Recognition**: Identifying specific individuals using deep learning embeddings

### Face Recognition Library

The `face-recognition` library provides state-of-the-art face recognition capabilities:

- **Deep Learning Based**: Uses a pre-trained deep neural network
- **Face Encoding**: Converts faces into 128-dimensional embeddings
- **High Accuracy**: More accurate than traditional methods
- **Face Comparison**: Calculate similarity between faces
- **Multiple Detection Models**: HOG (fast) and CNN (accurate) options

## Exercises

### Exercise 1: Basic Face Detection (`basic_face_detection.py`)

Learn fundamental face detection using OpenCV's Haar cascades:

- Load and configure Haar cascade classifiers
- Detect faces in static images
- Draw bounding boxes around detected faces
- Adjust detection parameters for better accuracy
- Handle multiple faces in a single image

### Exercise 2: Real-time Face Detection (`realtime_face_detection.py`)

Implement live face detection and recognition:

- Capture video from webcam
- Detect faces in real-time video streams
- Build a simple face database
- Perform face recognition against known faces
- Display confidence scores and person names

### Exercise 3: Advanced Face Recognition (`advanced_face_detection.py`)

Explore the face_recognition library for deep learning-based detection:

- Use deep learning models for more accurate face detection
- Compare Haar cascades vs. deep learning approaches
- Generate 128-dimensional face encodings
- Calculate face similarities and distances
- Understand encoding quality and robustness

### Exercise 4: Interactive Camera Face Recognition (`camera_face_capture.py`)

Complete face recognition workflow with camera integration:

- Interactive menu system for face recognition tasks
- Capture photos directly from camera
- Build and manage a face database
- Real-time face recognition with known faces
- Face analysis and similarity testing

### Exercise 5: Simple Camera Demo (`simple_camera_demo.py`)

Streamlined face recognition demonstration:

- Single-run photo capture and analysis
- Automatic face detection and annotation
- Educational information about face recognition
- Perfect for quick demonstrations and learning

### Exercise 3: Advanced Face Recognition Demo

Run `advanced_face_detection.py` to compare OpenCV Haar cascades with the face_recognition library and explore recognition accuracy.

### Exercise 4: Face Recognition Database

Learn how to build and manage a face recognition database with known individuals.

## Getting Started

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run basic face detection:

   ```bash
   python basic_face_detection.py
   ```

3. Try real-time face recognition (requires webcam):

   ```bash
   python realtime_face_detection.py
   ```

   All camera-based scripts accept a `--video-source` flag or the `VIDEO_SOURCE` environment variable. Pass a camera index (for example `0`) or a network stream URL such as `udp://host.docker.internal:5000`.

   ```bash
   # Use a network stream published from the host machine
   export VIDEO_SOURCE=udp://host.docker.internal:5000
   python realtime_face_detection.py

   # Or specify it per run
   python realtime_face_detection.py --video-source http://host.docker.internal:8080/stream.m3u8
   ```

   See `docs/remote-camera-streaming.md` for detailed platform-specific instructions on streaming webcams into containers.

4. Compare detection methods and explore face recognition:

   ```bash
   python advanced_face_detection.py
   ```

## Tips for Success

- Ensure good lighting conditions for better detection accuracy
- Experiment with different cascade files for various detection targets
- Understand the trade-off between detection accuracy and speed
- Try different scaling factors and minimum neighbor parameters
- Consider the computational requirements for real-time applications

## Common Challenges

- **False Positives**: Non-face objects detected as faces
- **Lighting Conditions**: Poor lighting affects detection accuracy
- **Pose Variations**: Profile views are harder to detect
- **Occlusions**: Partially hidden faces may not be detected
- **Scale Variations**: Very small or large faces may be missed

## Next Steps

Once you're comfortable with face detection, you'll move on to object detection using YOLO in Module 3!
