# RobotX Workshops: Computer Vision & AI

Welcome to the official code repository for the RobotX Computer Vision Workshop! This repo is designed to provide you with all the necessary code, resources, and project structures to master the fundamentals of computer vision and apply them to robotics technology.

## Workshop Overview

This workshop is a hands-on journey into the world of AI-powered robotics. We will move from the basic principles of image processing to advanced topics like real-time object detection and Simultaneous Localization and Mapping (SLAM). Each module is designed to be practical, building on the previous one.

### Learning Path

1. **Introduction to Computer Vision & Canny Edge Detection**: Learn the fundamentals of image processing.
2. **Face Detection**: Use pre-trained models to detect human faces in real-time.
3. **Object Detection with YOLO (Part 1)**: Get an introduction to the powerful YOLO algorithm.
4. **Object Detection with YOLO (Part 2)**: Fine-tune YOLO for custom object detection tasks.
5. **Visual SLAM**: Explore how robots can map and navigate their environment using computer vision.

## Requirements

### Python Version

This workshop is designed for **Python 3.8 to 3.11**. We recommend using Python 3.10 for the best compatibility with all dependencies.

**Note**: Python 3.12+ may have compatibility issues with some dependencies (particularly dlib and face-recognition). If you're using Python 3.12, you may need to install these packages using conda instead of pip.

### System Requirements

- **OS**: Windows 10+, macOS 10.14+, or Ubuntu 18.04+
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free disk space
- **Camera**: Optional (for real-time exercises, but not required)

## Getting Started

1. **Check Python Version**:
    Ensure you're using Python > 3.8:

    ```bash
    python --version
    ```

2. **Clone the Repository**:

    ```bash
    git clone [Your Repository URL]
    cd [Your Repository Name]
    ```

3. **Set Up Your Environment**:
    We recommend using a virtual environment to manage dependencies.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4. **Install Dependencies**:

    **Option A - Install all dependencies at once (recommended)**:

    ```bash
    pip install -r requirements.txt
    ```

    **Option B - Install module by module**:
    Navigate into each module's directory and install its specific requirements:

    ```bash
    cd 01-intro-canny-edge
    pip install -r requirements.txt
    cd ../02-face-detection
    pip install -r requirements.txt
    # ... continue for each module
    ```

    **Alternative for Python 3.12 users**: If you encounter dependency issues, try using conda:

    ```bash
    conda create -n computer-vision python=3.10
    conda activate computer-vision
    pip install -r requirements.txt
    ```

## Workshop Structure

The repository is divided into folders, one for each topic. Inside each folder, you will find:

- A `README.md` file with a detailed explanation of the topic.
- A Python script (`.py`) with example code.
- A `requirements.txt` file with the necessary libraries.

Let's start building the future of robotics, right here in Berlin!
