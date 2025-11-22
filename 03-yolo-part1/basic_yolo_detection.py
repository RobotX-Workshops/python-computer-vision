"""
Basic YOLO Detection - Module 3 Exercise 1
Learn fundamental object detection using YOLO models
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

class YOLODetector:
    """
    Object detection using YOLO models
    """
    
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights
        """
        try:
            self.model = YOLO(model_path)
            print(f"YOLO model loaded successfully: {model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Downloading YOLOv8 nano model...")
            self.model = YOLO('yolov8n.pt')  # This will auto-download
        
        # COCO class names (80 classes)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def detect_objects(self, image, confidence_threshold=0.5, iou_threshold=0.5):
        """
        Detect objects in an image
        
        Args:
            image: Input image
            confidence_threshold: Minimum confidence for detection
            iou_threshold: IoU threshold for NMS
        
        Returns:
            results: YOLO detection results
            annotated_image: Image with detection annotations
        """
        # Run inference
        results = self.model(image, conf=confidence_threshold, iou=iou_threshold)
        
        # Get annotated image
        annotated_image = results[0].plot()
        
        return results[0], annotated_image
    
    def get_detection_info(self, results):
        """
        Extract detection information from results
        
        Args:
            results: YOLO detection results
        
        Returns:
            detections: List of detection dictionaries
        """
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                detection = {
                    'bbox': boxes[i],
                    'confidence': confidences[i],
                    'class_id': class_ids[i],
                    'class_name': self.class_names[class_ids[i]],
                }
                detections.append(detection)
        
        return detections

def create_test_scene():
    """
    Create a test scene with common objects
    """
    # Create a simple scene with geometric shapes representing objects
    scene = np.ones((600, 800, 3), dtype=np.uint8) * 240
    
    # Draw objects that might be detected
    # Car-like rectangle
    cv2.rectangle(scene, (100, 400), (300, 550), (0, 0, 200), -1)
    cv2.rectangle(scene, (120, 500), (140, 530), (0, 0, 0), -1)  # Wheel
    cv2.rectangle(scene, (260, 500), (280, 530), (0, 0, 0), -1)  # Wheel
    
    # Person-like figure
    cv2.circle(scene, (500, 200), 40, (150, 100, 50), -1)  # Head
    cv2.rectangle(scene, (480, 240), (520, 400), (150, 100, 50), -1)  # Body
    cv2.rectangle(scene, (460, 320), (480, 450), (150, 100, 50), -1)  # Left arm
    cv2.rectangle(scene, (520, 320), (540, 450), (150, 100, 50), -1)  # Right arm
    cv2.rectangle(scene, (485, 400), (505, 500), (150, 100, 50), -1)  # Left leg
    cv2.rectangle(scene, (505, 400), (525, 500), (150, 100, 50), -1)  # Right leg
    
    # Bottle-like object
    cv2.rectangle(scene, (600, 300), (620, 450), (0, 150, 0), -1)
    cv2.rectangle(scene, (605, 280), (615, 300), (0, 150, 0), -1)  # Neck
    
    return scene

def demonstrate_confidence_threshold():
    """
    Demonstrate effect of different confidence thresholds
    """
    print("=== Confidence Threshold Demo ===")
    
    detector = YOLODetector()
    
    # Load existing test image
    test_image_path = 'yolo_test_scene.jpeg'
    if not os.path.exists(test_image_path):
        print(f"Error: {test_image_path} not found. Creating a simple test scene...")
        test_image = create_test_scene()
    else:
        test_image = cv2.imread(test_image_path)
        if test_image is None:
            print(f"Error: Could not load {test_image_path}. Creating a simple test scene...")
            test_image = create_test_scene()
        else:
            print(f"Loaded test image: {test_image_path}")
    
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    plt.figure(figsize=(20, 12))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Test Scene')
    plt.axis('off')
    
    for i, threshold in enumerate(thresholds):
        results, annotated = detector.detect_objects(test_image, confidence_threshold=threshold)
        detections = detector.get_detection_info(results)
        
        plt.subplot(2, 3, i + 2)
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.title(f'Confidence ≥ {threshold}\nDetections: {len(detections)}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_detections():
    """
    Analyze detection results with different images
    """
    print("=== Detection Analysis ===")
    
    detector = YOLODetector()
    
    # Load existing test image
    test_image_path = 'yolo_test_scene.jpeg'
    if os.path.exists(test_image_path):
        test_images = [cv2.imread(test_image_path)]
        print(f"Using existing test image: {test_image_path}")
    else:
        # Fallback to created scene if file doesn't exist
        test_images = [create_test_scene()]
        print("Test image not found, using generated scene")
    
    for i, image in enumerate(test_images):
        if image is None:
            print(f"Skipping test image {i + 1} - image is None")
            continue
            
        print(f"\nAnalyzing test image {i + 1}...")
        
        results, annotated = detector.detect_objects(image)
        detections = detector.get_detection_info(results)
        
        print(f"Total detections: {len(detections)}")
        
        # Group by class
        class_counts = {}
        for detection in detections:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("Objects detected:")
        for class_name, count in class_counts.items():
            avg_conf = np.mean([d['confidence'] for d in detections if d['class_name'] == class_name])
            print(f"  {class_name}: {count} (avg confidence: {avg_conf:.2f})")
        
        # Display results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'Test Image {i + 1}')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.title(f'YOLO Detections\nTotal: {len(detections)} objects')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function demonstrating YOLO object detection
    """
    print("=== YOLO Object Detection Demo ===")
    
    try:
        # Initialize detector
        detector = YOLODetector()
        
        print("1. Using existing test scene...")
        if not os.path.exists('yolo_test_scene.jpeg'):
            print("   yolo_test_scene.jpeg not found, creating fallback scene...")
            test_scene = create_test_scene()
            cv2.imwrite('yolo_test_scene.png', test_scene)
        else:
            print("   Found yolo_test_scene.jpeg")
        
        print("2. Demonstrating confidence thresholds...")
        demonstrate_confidence_threshold()
        
        print("3. Analyzing detections...")
        analyze_detections()
        
        print("\n=== Exercise Complete ===")
        print("Key Learnings:")
        print("- YOLO can detect multiple objects in a single pass")
        print("- Confidence threshold affects detection sensitivity")
        print("- Higher thresholds reduce false positives but may miss objects")
        print("- YOLO provides both bounding boxes and class probabilities")
        print("- Modern YOLO versions are very fast and accurate")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Make sure you have installed the required dependencies:")
        print("pip install ultralytics torch torchvision")

if __name__ == "__main__":
    main()