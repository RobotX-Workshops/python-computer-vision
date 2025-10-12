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
- **Face Recognition**: Identifying specific individuals

## Exercises

### Exercise 1: Basic Face Detection

Run `basic_face_detection.py` to learn fundamental face detection with Haar cascades.

### Exercise 2: Real-time Face Detection

Run `realtime_face_detection.py` to detect faces from webcam feed in real-time.

### Exercise 3: Advanced Face Detection

Run `advanced_face_detection.py` to explore DNN-based face detection and landmarks.

### Exercise 4: Multiple Face Detection

Run `multi_face_detection.py` to handle scenarios with multiple faces.

## Getting Started

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run basic face detection:

   ```bash
   python basic_face_detection.py
   ```

3. Try real-time detection (requires webcam):

   ```bash
   python realtime_face_detection.py
   ```

4. Explore advanced features:

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
