"""
Advanced Face Detection with Recognition - Module 2 Exercise 3
Compare OpenCV Haar cascades with face_recognition library capabilities
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import face_recognition
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

class FaceRecognitionDemo:
    """
    Demonstrate face recognition capabilities using the face_recognition library
    """
    
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
    
    def load_and_encode_face(self, image_path, person_name):
        """
        Load an image and create face encoding
        
        Args:
            image_path: Path to the image file
            person_name: Name to associate with the face
        
        Returns:
            success: Boolean indicating if encoding was successful
        """
        try:
            image_path = Path(image_path)
            if not image_path.is_absolute():
                image_path = DATA_DIR / image_path

            # Load image
            image = face_recognition.load_image_file(str(image_path))
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                # Use the first face found
                face_encoding = face_encodings[0]
                
                # Add to known faces
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(person_name)
                
                print(f"Successfully encoded face for {person_name}")
                return True
            else:
                print(f"No face found in {image_path}")
                return False
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False
    
    def create_sample_database(self):
        """
        Create a sample face database using generated images
        """
        # Create sample face images using shapes (for demonstration)
        sample_faces = []
        
        # Sample face 1 - circular face
        face1 = np.ones((200, 200, 3), dtype=np.uint8) * 220
        cv2.circle(face1, (100, 100), 80, (180, 160, 140), -1)  # Face
        cv2.circle(face1, (80, 80), 8, (0, 0, 0), -1)   # Left eye
        cv2.circle(face1, (120, 80), 8, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(face1, (100, 120), (15, 8), 0, 0, 180, (100, 50, 50), 2)  # Mouth
        alice_path = DATA_DIR / 'sample_person_alice.jpg'
        cv2.imwrite(str(alice_path), face1)
        sample_faces.append((alice_path, 'Alice'))
        
        # Sample face 2 - oval face
        face2 = np.ones((200, 200, 3), dtype=np.uint8) * 210
        cv2.ellipse(face2, (100, 100), (75, 90), 0, 0, 360, (190, 170, 150), -1)  # Face
        cv2.circle(face2, (85, 85), 6, (0, 0, 0), -1)   # Left eye
        cv2.circle(face2, (115, 85), 6, (0, 0, 0), -1)  # Right eye
        cv2.rectangle(face2, (95, 115), (105, 125), (120, 60, 60), -1)  # Nose
        cv2.ellipse(face2, (100, 135), (12, 6), 0, 0, 180, (100, 50, 50), 2)  # Mouth
        bob_path = DATA_DIR / 'sample_person_bob.jpg'
        cv2.imwrite(str(bob_path), face2)
        sample_faces.append((bob_path, 'Bob'))
        
        print("Created sample face images:")
        for filepath, name in sample_faces:
            print(f"  {filepath.name} -> {name}")
        
        return sample_faces
    
    def recognize_faces_in_image(self, test_image_path):
        """
        Recognize faces in a test image
        
        Args:
            test_image_path: Path to the test image
        
        Returns:
            results: Dictionary containing recognition results
        """
        try:
            # Load test image
            test_image_path = Path(test_image_path)
            if not test_image_path.is_absolute():
                test_image_path = DATA_DIR / test_image_path

            test_image = face_recognition.load_image_file(str(test_image_path))
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(test_image)
            face_encodings = face_recognition.face_encodings(test_image, face_locations)
            
            results = {
                'image': test_image,
                'face_locations': face_locations,
                'recognized_names': [],
                'confidence_scores': []
            }
            
            # Compare each face with known faces
            for face_encoding in face_encodings:
                if len(self.known_face_encodings) > 0:
                    # Calculate distances to all known faces
                    distances = face_recognition.face_distance(
                        self.known_face_encodings, 
                        face_encoding
                    )
                    
                    # Find the best match
                    best_match_index = np.argmin(distances)
                    confidence = 1 - distances[best_match_index]
                    
                    # Determine if it's a good match (threshold: 0.4)
                    if distances[best_match_index] < 0.6:
                        name = self.known_face_names[best_match_index]
                    else:
                        name = "Unknown"
                    
                    results['recognized_names'].append(name)
                    results['confidence_scores'].append(confidence)
                else:
                    results['recognized_names'].append("Unknown")
                    results['confidence_scores'].append(0.0)
            
            return results
            
        except Exception as e:
            print(f"Error recognizing faces in {test_image_path}: {e}")
            return None
    
    def compare_detection_methods(self, image_path):
        """
        Compare OpenCV Haar cascades with face_recognition library
        
        Args:
            image_path: Path to test image
        """
        # Load image with OpenCV
        image_path = Path(image_path)
        if not image_path.is_absolute():
            image_path = DATA_DIR / image_path

        opencv_image = cv2.imread(str(image_path))
        if opencv_image is None:
            print(f"Could not load image: {image_path}")
            return
        
        # Method 1: OpenCV Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        opencv_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Method 2: face_recognition library
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(rgb_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # OpenCV detection
        opencv_display = rgb_image.copy()
        for (x, y, w, h) in opencv_faces:
            cv2.rectangle(opencv_display, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        axes[1].imshow(opencv_display)
        axes[1].set_title(f'OpenCV Haar Cascade\n({len(opencv_faces)} faces detected)')
        axes[1].axis('off')
        
        # face_recognition detection
        fr_display = rgb_image.copy()
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(fr_display, (left, top), (right, bottom), (0, 255, 0), 2)
        
        axes[2].imshow(fr_display)
        axes[2].set_title(f'face_recognition Library\n({len(face_locations)} faces detected)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return len(opencv_faces), len(face_locations)
    
    def demonstrate_recognition_accuracy(self):
        """
        Demonstrate face recognition accuracy with different similarity levels
        """
        if len(self.known_face_encodings) < 2:
            print("Need at least 2 known faces to demonstrate recognition accuracy")
            return
        
        print("=== Face Recognition Accuracy Demonstration ===")
        
        # Test recognition with different tolerance levels
        tolerance_levels = [0.4, 0.5, 0.6, 0.7, 0.8]
        
        print(f"Testing with {len(self.known_face_names)} known faces: {', '.join(self.known_face_names)}")
        print("\nTolerance Level | Strict/Loose Recognition")
        print("-" * 45)
        
        for tolerance in tolerance_levels:
            # Calculate average distance between different people
            if len(self.known_face_encodings) >= 2:
                distances = []
                for i in range(len(self.known_face_encodings)):
                    for j in range(i + 1, len(self.known_face_encodings)):
                        dist = face_recognition.face_distance(
                            [self.known_face_encodings[i]], 
                            self.known_face_encodings[j]
                        )[0]
                        distances.append(dist)
                
                avg_distance = np.mean(distances)
                
                if tolerance < avg_distance:
                    recognition_type = "Strict (fewer false positives)"
                else:
                    recognition_type = "Loose (more false positives)"
                
                print(f"{tolerance:13.1f} | {recognition_type}")

def main():
    """
    Main demonstration of face recognition capabilities
    """
    print("=== Advanced Face Recognition Demo ===")
    
    # Initialize face recognition system
    face_recognizer = FaceRecognitionDemo()
    
    # Check for existing face images
    existing_faces = [
        path for path in DATA_DIR.glob('*')
        if path.suffix.lower() in {'.jpg', '.jpeg', '.png'} and 'person' in path.name.lower()
    ]
    
    if not existing_faces:
        print("No existing face images found. Creating sample database...")
        sample_faces = face_recognizer.create_sample_database()
        
        # Load the sample faces
        for filepath, name in sample_faces:
            face_recognizer.load_and_encode_face(filepath, name)

        existing_faces = [filepath for filepath, _ in sample_faces]
    else:
        print(f"Found existing face images: {[path.name for path in existing_faces]}")
        # Load existing faces
        for filename in existing_faces:
            # Extract name from filename
            name = filename.name.replace('sample_person_', '').replace('.jpg', '').replace('.jpeg', '').replace('.png', '').title()
            face_recognizer.load_and_encode_face(filename, name)
    
    if len(face_recognizer.known_face_names) == 0:
        print("No faces could be encoded. Please check your image files.")
        return
    
    print(f"\nLoaded {len(face_recognizer.known_face_names)} known faces:")
    for i, name in enumerate(face_recognizer.known_face_names):
        print(f"  {i + 1}. {name}")
    
    # Demonstrate recognition accuracy
    face_recognizer.demonstrate_recognition_accuracy()
    
    # Test recognition on known images
    print("\n=== Testing Recognition ===")
    for i, filename in enumerate(existing_faces[:2]):  # Test first 2 images
        print(f"\nTesting recognition on {filename.name}...")
        results = face_recognizer.recognize_faces_in_image(filename)
        
        if results:
            print(f"Found {len(results['face_locations'])} face(s):")
            for j, (name, confidence) in enumerate(zip(results['recognized_names'], results['confidence_scores'])):
                print(f"  Face {j + 1}: {name} (confidence: {confidence:.2f})")
    
    # Compare detection methods
    if existing_faces:
        print("\n=== Comparing Detection Methods ===")
        test_image = existing_faces[0]
        comparison = face_recognizer.compare_detection_methods(test_image)
        if comparison:
            opencv_count, fr_count = comparison

            print(f"Detection comparison on {test_image.name}:")
            print(f"  OpenCV Haar Cascade: {opencv_count} faces")
            print(f"  face_recognition library: {fr_count} faces")
    
    print("\n=== Demo Complete ===")
    print("Key Learnings:")
    print("- face_recognition library provides more accurate detection than Haar cascades")
    print("- Face recognition uses deep learning embeddings for identification")
    print("- Tolerance levels control the trade-off between false positives and false negatives")
    print("- Real face recognition requires good quality reference images")

if __name__ == "__main__":
    main()