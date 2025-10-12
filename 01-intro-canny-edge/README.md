# Module 1: Introduction to Computer Vision & Canny Edge Detection

## Overview

Welcome to your first hands-on experience with computer vision! In this module, you'll learn the fundamentals of image processing and implement the famous Canny edge detection algorithm. Edge detection is one of the most fundamental techniques in computer vision and forms the basis for many more advanced algorithms.

## Learning Objectives

By the end of this module, you will:

- Understand basic image processing concepts
- Learn about different types of image filters and their effects
- Implement the Canny edge detection algorithm
- Understand the parameters that affect edge detection quality
- Process images from files and webcam feeds

## Theory Background

### What is Edge Detection?

Edge detection is a fundamental technique in computer vision that identifies boundaries of objects within images. Edges occur where there are significant changes in pixel intensity, which typically correspond to:

- Object boundaries
- Surface orientation changes
- Material property changes
- Illumination changes

### The Canny Edge Detection Algorithm

Developed by John Canny in 1986, the Canny edge detector is considered one of the optimal edge detection algorithms. It follows these steps:

1. **Noise Reduction**: Apply Gaussian filter to reduce noise
2. **Gradient Calculation**: Find intensity gradients using Sobel filters
3. **Non-Maximum Suppression**: Thin edges to single-pixel width
4. **Double Thresholding**: Identify strong and weak edges
5. **Edge Tracking**: Connect weak edges to strong edges

### Parameters

- **Low Threshold**: Minimum gradient magnitude for weak edges
- **High Threshold**: Minimum gradient magnitude for strong edges
- **Gaussian Kernel Size**: Size of the noise reduction filter

## Exercises

### Exercise 1: Basic Image Loading and Display

Run `basic_image_ops.py` to learn how to load, display, and manipulate images.

### Exercise 2: Canny Edge Detection

Run `canny_edge_detection.py` to implement and experiment with Canny edge detection.

### Exercise 3: Real-time Edge Detection

Run `realtime_canny.py` to apply edge detection to webcam feed in real-time.

### Exercise 4: Parameter Tuning

Experiment with different threshold values and observe their effects on edge detection quality.

## Getting Started

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the basic image operations script:

   ```bash
   python basic_image_ops.py
   ```

3. Try the Canny edge detection:

   ```bash
   python canny_edge_detection.py
   ```

4. For real-time detection (requires webcam):

   ```bash
   python realtime_canny.py
   ```

## Tips for Success

- Start with the provided sample images before using your own
- Experiment with different threshold values to see their effects
- Try the algorithm on different types of images (portraits, landscapes, objects)
- Pay attention to the relationship between noise and edge detection quality

## Next Steps

Once you're comfortable with edge detection, you'll move on to face detection using pre-trained models in Module 2!
