"""
Basic Face Detection - Module 2 Exercise 1
Learn fundamental face detection using Haar cascade classifiers
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class FaceDetector:
    """
    Basic face detection using Haar cascade classifiers
    """
    
    def __init__(self):
        # Load pre-trained Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Check if cascades loaded successfully
        if self.face_cascade.empty():
            print("Error: Could not load face cascade classifier")
        if self.eye_cascade.empty():
            print("Error: Could not load eye cascade classifier")
        if self.smile_cascade.empty():
            print("Error: Could not load smile cascade classifier")
    
    def detect_faces(self, image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detect faces in an image using Haar cascade
        
        Args:
            image: Input image (grayscale or color)
            scale_factor: How much the image size is reduced at each scale
            min_neighbors: How many neighbors each face should have to retain it
            min_size: Minimum possible face size (smaller faces are ignored)
        
        Returns:
            faces: Array of rectangles where faces were found
            gray: Grayscale version of input image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces, gray
    
    def detect_eyes_and_smile(self, gray, face_roi):
        """
        Detect eyes and smile within a face region
        
        Args:
            gray: Grayscale image
            face_roi: Face region of interest (x, y, w, h)
        
        Returns:
            eyes: Array of eye rectangles
            smiles: Array of smile rectangles
        """
        x, y, w, h = face_roi
        roi_gray = gray[y:y+h, x:x+w]
        
        # Detect eyes in the upper half of face
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray[0:h//2, 0:w],  # Upper half only
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(10, 10)
        )
        
        # Detect smile in the lower half of face
        smiles = self.smile_cascade.detectMultiScale(
            roi_gray[h//2:h, 0:w],  # Lower half only
            scaleFactor=1.7,
            minNeighbors=15,
            minSize=(15, 15)
        )
        
        # Adjust coordinates to original image
        if len(eyes) > 0:
            eyes[:, 0] += x  # Adjust x coordinate
            eyes[:, 1] += y  # Adjust y coordinate
        
        if len(smiles) > 0:
            smiles[:, 0] += x  # Adjust x coordinate  
            smiles[:, 1] += y + h//2  # Adjust y coordinate (add offset for lower half)
        
        return eyes, smiles
    
    def draw_detections(self, image, faces, eyes=None, smiles=None):
        """
        Draw detection rectangles on image
        
        Args:
            image: Input image
            faces: Array of face rectangles
            eyes: Array of eye rectangles (optional)
            smiles: Array of smile rectangles (optional)
        
        Returns:
            annotated_image: Image with detection rectangles drawn
        """
        annotated = image.copy()
        
        # Draw face rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(annotated, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw eye rectangles
        if eyes is not None:
            for (x, y, w, h) in eyes:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(annotated, 'Eye', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw smile rectangles
        if smiles is not None:
            for (x, y, w, h) in smiles:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated, 'Smile', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return annotated

def create_test_images():
    """
    Create synthetic test images with face-like patterns
    """
    # Test image 1: Simple face-like pattern
    img1 = np.ones((300, 300, 3), dtype=np.uint8) * 200
    
    # Draw a simple face
    cv2.ellipse(img1, (150, 150), (80, 100), 0, 0, 360, (180, 180, 180), -1)  # Face outline
    cv2.circle(img1, (130, 130), 10, (0, 0, 0), -1)  # Left eye
    cv2.circle(img1, (170, 130), 10, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(img1, (150, 180), (20, 10), 0, 0, 180, (0, 0, 0), 2)  # Smile
    
    # Test image 2: Multiple face-like patterns
    img2 = np.ones((400, 600, 3), dtype=np.uint8) * 220
    
    # Draw multiple simple faces
    faces_positions = [(150, 150), (450, 150), (300, 280)]
    for cx, cy in faces_positions:
        cv2.ellipse(img2, (cx, cy), (60, 75), 0, 0, 360, (160, 160, 160), -1)
        cv2.circle(img2, (cx - 20, cy - 20), 8, (0, 0, 0), -1)
        cv2.circle(img2, (cx + 20, cy - 20), 8, (0, 0, 0), -1)
        cv2.ellipse(img2, (cx, cy + 25), (15, 8), 0, 0, 180, (0, 0, 0), 2)
    
    return img1, img2

def demonstrate_parameters():
    """
    Demonstrate how different parameters affect face detection
    """
    print("=== Parameter Effects Demonstration ===")
    
    # Create test image
    test_img, _ = create_test_images()
    
    detector = FaceDetector()
    
    # Different parameter combinations
    parameter_sets = [
        (1.05, 3, (20, 20), "Very Sensitive"),
        (1.1, 5, (30, 30), "Default"),
        (1.3, 7, (50, 50), "Conservative"),
        (1.1, 10, (30, 30), "High Min Neighbors")
    ]
    
    plt.figure(figsize=(20, 10))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    for i, (scale_factor, min_neighbors, min_size, label) in enumerate(parameter_sets):
        faces, gray = detector.detect_faces(
            test_img, 
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
            min_size=min_size
        )
        
        annotated = detector.draw_detections(test_img, faces)
        
        plt.subplot(2, 3, i + 2)
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.title(f'{label}\nSF:{scale_factor} MN:{min_neighbors} MS:{min_size}\nFaces: {len(faces)}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def comprehensive_detection_demo():
    """
    Comprehensive demonstration of face, eye, and smile detection
    """
    print("=== Comprehensive Detection Demo ===")
    
    detector = FaceDetector()
    
    # Create test images
    test_img1, test_img2 = create_test_images()
    test_images = [test_img1, test_img2]
    
    for i, img in enumerate(test_images):
        print(f"\nProcessing test image {i + 1}...")
        
        # Detect faces
        faces, gray = detector.detect_faces(img)
        print(f"Found {len(faces)} faces")
        
        # Detect eyes and smiles for each face
        all_eyes = []
        all_smiles = []
        
        for face in faces:
            eyes, smiles = detector.detect_eyes_and_smile(gray, face)
            if len(eyes) > 0:
                all_eyes.extend(eyes)
            if len(smiles) > 0:
                all_smiles.extend(smiles)
        
        print(f"Found {len(all_eyes)} eyes")
        print(f"Found {len(all_smiles)} smiles")
        
        # Create annotated image
        annotated = detector.draw_detections(img, faces, all_eyes, all_smiles)
        
        # Display results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Original Image {i + 1}')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.title(f'Detections\nFaces: {len(faces)}, Eyes: {len(all_eyes)}, Smiles: {len(all_smiles)}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Save annotated image
        cv2.imwrite(f'face_detection_result_{i + 1}.png', annotated)

def detection_statistics(detector, image):
    """
    Analyze detection statistics for different parameter settings
    """
    print("=== Detection Statistics Analysis ===")
    
    scale_factors = [1.05, 1.1, 1.2, 1.3]
    min_neighbors_values = [3, 5, 7, 10]
    
    results = []
    
    for sf in scale_factors:
        for mn in min_neighbors_values:
            faces, _ = detector.detect_faces(image, scale_factor=sf, min_neighbors=mn)
            results.append({
                'scale_factor': sf,
                'min_neighbors': mn,
                'face_count': len(faces),
                'faces': faces
            })
    
    # Display statistics
    print(f"{'Scale Factor':<12} {'Min Neighbors':<12} {'Face Count':<10}")
    print("-" * 34)
    
    for result in results:
        print(f"{result['scale_factor']:<12} {result['min_neighbors']:<12} {result['face_count']:<10}")
    
    # Find optimal parameters (most common face count)
    face_counts = [r['face_count'] for r in results]
    most_common_count = max(set(face_counts), key=face_counts.count)
    
    optimal_results = [r for r in results if r['face_count'] == most_common_count]
    
    print(f"\nMost common face count: {most_common_count}")
    print("Optimal parameter combinations:")
    for result in optimal_results:
        print(f"  Scale Factor: {result['scale_factor']}, Min Neighbors: {result['min_neighbors']}")
    
    return results

def main():
    """
    Main function demonstrating basic face detection
    """
    print("=== Basic Face Detection Demo ===")
    
    # Create detector
    detector = FaceDetector()
    
    # Check if cascades loaded properly
    if detector.face_cascade.empty():
        print("Error: Face cascade not loaded. Please check OpenCV installation.")
        return
    
    print("1. Creating test images...")
    test_img1, test_img2 = create_test_images()
    
    # Save test images
    cv2.imwrite('test_face_1.png', test_img1)
    cv2.imwrite('test_face_2.png', test_img2)
    
    print("2. Demonstrating parameter effects...")
    demonstrate_parameters()
    
    print("3. Running comprehensive detection demo...")
    comprehensive_detection_demo()
    
    print("4. Analyzing detection statistics...")
    detection_statistics(detector, test_img2)
    
    print("\n=== Exercise Complete ===")
    print("Key Learnings:")
    print("- Haar cascades work well for frontal face detection")
    print("- Scale factor controls detection thoroughness vs speed")
    print("- Min neighbors reduces false positives")
    print("- Minimum size prevents detection of very small faces")
    print("- Eye and smile detection requires face-relative coordinates")
    print("- Parameter tuning is crucial for optimal results")

if __name__ == "__main__":
    main()