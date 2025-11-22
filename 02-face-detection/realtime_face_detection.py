"""
Real-time Face Detection and Recognition - Module 2 Exercise 2
Advanced face detection using both OpenCV Haar cascades and face_recognition library
"""

import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import face_recognition
import numpy as np


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

class AdvancedFaceDetector:
    """
    Advanced face detection and recognition using multiple methods
    """
    
    def __init__(self):
        # OpenCV Haar cascade for fast detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' # type: ignore
        )
        
        # Known faces database
        self.known_face_encodings = []
        self.known_face_names = []
        self.faces_db_path = DATA_DIR / "known_faces.pkl"
        
        # Load existing face database if it exists
        self.load_face_database()
        
        # Detection parameters
        self.face_detection_method = "hog"  # "hog" or "cnn"
        self.recognition_tolerance = 0.6
        
    def load_face_database(self):
        """Load known faces from pickle file"""
        if self.faces_db_path.exists():
            try:
                with self.faces_db_path.open('rb') as f:
                    data = pickle.load(f)
                    print("The data from the face database is as follows:")
                    print(data)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"Loaded {len(self.known_face_names)} known faces from database")
            except Exception as e:
                print(f"Error loading face database: {e}")
        else:
            print("No existing face database found. Starting fresh.")
    
    def save_face_database(self):
        """Save known faces to pickle file"""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }
            with self.faces_db_path.open('wb') as f:
                pickle.dump(data, f)
            print(f"Saved {len(self.known_face_names)} faces to database")
        except Exception as e:
            print(f"Error saving face database: {e}")
    
    def add_known_face(self, face_image, name):
        """
        Add a new face to the known faces database
        
        Args:
            face_image: Image containing the face to add
            name: Name to associate with the face
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_image)
        
        if len(face_encodings) > 0:
            # Use the first face found
            face_encoding = face_encodings[0]
            
            # Check if this person already exists
            if name in self.known_face_names:
                # Update existing encoding
                idx = self.known_face_names.index(name)
                self.known_face_encodings[idx] = face_encoding
                print(f"Updated face encoding for {name}")
            else:
                # Add new face
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)
                print(f"Added new face: {name}")
            
            # Save to database
            self.save_face_database()
            return True
        else:
            print(f"No face found in image for {name}")
            return False
    
    def detect_faces_opencv(self, frame):
        """Fast face detection using OpenCV Haar cascades"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def detect_and_recognize_faces(self, frame):
        """
        Detect and recognize faces using face_recognition library
        
        Args:
            frame: Input image frame
            
        Returns:
            face_locations: List of face locations
            face_names: List of recognized names
            face_confidences: List of confidence scores
        """
        # Convert BGR to RGB for face_recognition library
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(
            rgb_frame, 
            model=self.face_detection_method
        )
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        face_names = []
        face_confidences = []
        
        for face_encoding in face_encodings:
            # Compare with known faces
            if len(self.known_face_encodings) > 0:
                distances = face_recognition.face_distance(
                    self.known_face_encodings, 
                    face_encoding
                )
                
                best_match_index = np.argmin(distances)
                confidence = 1 - distances[best_match_index]
                
                if distances[best_match_index] < self.recognition_tolerance:
                    name = self.known_face_names[best_match_index]
                    face_names.append(name)
                    face_confidences.append(confidence)
                else:
                    face_names.append("Unknown")
                    face_confidences.append(confidence)
            else:
                face_names.append("Unknown")
                face_confidences.append(0.0)
        
        return face_locations, face_names, face_confidences
    
    def draw_face_annotations(self, frame, face_locations, face_names, face_confidences):
        """
        Draw face detection and recognition annotations
        
        Args:
            frame: Input image
            face_locations: List of (top, right, bottom, left) tuples
            face_names: List of recognized names
            face_confidences: List of confidence scores
            
        Returns:
            annotated_frame: Frame with annotations
        """
        annotated_frame = frame.copy()
        
        for (top, right, bottom, left), name, confidence in zip(
            face_locations, face_names, face_confidences
        ):
            # Choose color based on recognition status
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
                label = f"Unknown ({confidence:.2f})"
            else:
                color = (0, 255, 0)  # Green for known
                label = f"{name} ({confidence:.2f})"
            
            # Draw rectangle around face
            cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            cv2.rectangle(annotated_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Draw label text
            cv2.putText(
                annotated_frame, label, (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1
            )
        
        return annotated_frame
    
    def create_comparison_view(self, frame):
        """Create side-by-side comparison of detection methods"""
        # Method 1: OpenCV Haar cascades (fast)
        opencv_faces = self.detect_faces_opencv(frame)
        opencv_frame = frame.copy()
        
        for (x, y, w, h) in opencv_faces:
            cv2.rectangle(opencv_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(opencv_frame, 'Haar Cascade', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Method 2: face_recognition library (accurate)
        face_locations, face_names, face_confidences = self.detect_and_recognize_faces(frame)
        recognition_frame = self.draw_face_annotations(frame, face_locations, face_names, face_confidences)
        
        # Add method labels
        cv2.putText(opencv_frame, 'OpenCV Haar Cascade (Fast)', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(recognition_frame, 'Face Recognition Library (Accurate)', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Combine frames side by side
        combined = np.hstack([opencv_frame, recognition_frame])
        
        return combined, len(opencv_faces), len(face_locations)

def add_faces_from_images():
    """Add known faces from image files"""
    detector = AdvancedFaceDetector()
    
    # Look for face images in the shared data directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    face_images = [
        path for path in DATA_DIR.glob('*')
        if path.suffix.lower() in image_extensions and (
            'face' in path.name.lower() or 'person' in path.name.lower()
        )
    ]
    
    if not face_images:
        print("No face images found. Add files containing 'face' or 'person' in their name to the data/ directory.")
        return detector
    
    print(f"Found potential face images: {[path.name for path in face_images]}")
    
    for image_file in face_images:
        try:
            # Load image
            image = cv2.imread(str(image_file))
            if image is None:
                continue
                
            # Extract name from filename (remove extension and common prefixes)
            name = image_file.stem
            name = name.replace('face_', '').replace('person_', '').replace('_', ' ').title()
            
            # Add to known faces
            success = detector.add_known_face(image, name)
            if success:
                print(f"Successfully added {name} from {image_file.name}")
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
    
    return detector

def create_sample_faces():
    """Create sample face images for testing"""
    print("Creating sample face images for testing...")
    
    # You can add code here to download sample face images
    # For now, we'll just create placeholder instructions
    
    sample_faces = [
        ("sample_person1.jpg", "Person 1"),
        ("sample_person2.jpg", "Person 2"),
    ]
    
    print("To test face recognition:")
    print(f"1. Add some face images to {DATA_DIR} (inside the repository)")
    print("2. Name them with 'face_' or 'person_' prefix")
    print("3. Run this script again")
    print("\nExample filenames:")
    for filename, name in sample_faces:
        print(f"  {filename} -> Will be recognized as '{name}'")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Real-time face detection with optional network camera support"
    )
    parser.add_argument(
        "--video-source",
        default=None,
        help="Camera index (e.g. 0) or network stream URL (e.g. udp://host.docker.internal:5000)"
    )
    return parser.parse_args()


def resolve_video_source(cli_value: Optional[str]) -> str:
    """Resolve the active video source from CLI or environment"""
    env_value = os.getenv("VIDEO_SOURCE")
    if cli_value:
        return cli_value
    if env_value:
        return env_value
    return "0"


def open_capture(source: str) -> cv2.VideoCapture:
    """Create a cv2.VideoCapture using either index or URL"""
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def main():
    """Main function for real-time face detection and recognition"""
    print("=== Advanced Face Detection and Recognition ===")

    args = parse_args()
    video_source = resolve_video_source(args.video_source)
    print(f"Video source: {video_source}")

    # Initialize detector
    detector = AdvancedFaceDetector()

    # Try to add faces from existing images
    detector = add_faces_from_images()

    # If no known faces, create samples
    if len(detector.known_face_names) == 0:
        create_sample_faces()
        print("\nContinuing with detection only (no recognition)...")

    # Initialize camera
    cap = open_capture(video_source)
    if not cap.isOpened():
        print("Error: Cannot access camera. Using sample images instead.")
        return
    
    print("\n=== Controls ===")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("Press 'a' to add current face to database")
    print("Press 'c' to toggle comparison mode")
    print("Press 'm' to toggle detection method (HOG/CNN)")
    print("Press 'r' to reset face database")
    
    comparison_mode = False
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if comparison_mode:
                # Show comparison of both methods
                display_frame, opencv_count, recognition_count = detector.create_comparison_view(frame)
                
                # Add statistics
                stats_text = f"Frame: {frame_count} | OpenCV: {opencv_count} faces | face_recognition: {recognition_count} faces"
                cv2.putText(display_frame, stats_text, (10, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # Show only face_recognition method
                face_locations, face_names, face_confidences = detector.detect_and_recognize_faces(frame)
                display_frame = detector.draw_face_annotations(frame, face_locations, face_names, face_confidences)
                
                # Add statistics
                stats_text = f"Frame: {frame_count} | Faces: {len(face_locations)} | Known: {len(detector.known_face_names)}"
                cv2.putText(display_frame, stats_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Advanced Face Detection', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = DATA_DIR / f"face_detection_frame_{frame_count}_{datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(str(filename), display_frame)
                print(f"Saved frame as {filename}")
            elif key == ord('a'):
                # Add current face to database
                name = input("\nEnter name for this person (or press Enter to cancel): ").strip()
                if name:
                    success = detector.add_known_face(frame, name)
                    if success:
                        print(f"Added {name} to face database!")
                    else:
                        print("No clear face found in current frame")
            elif key == ord('c'):
                comparison_mode = not comparison_mode
                print(f"Comparison mode: {'ON' if comparison_mode else 'OFF'}")
            elif key == ord('m'):
                detector.face_detection_method = "cnn" if detector.face_detection_method == "hog" else "hog"
                print(f"Detection method: {detector.face_detection_method.upper()}")
            elif key == ord('r'):
                # Reset face database
                confirm = input("\nReset face database? This will delete all known faces. (y/N): ").strip().lower()
                if confirm == 'y':
                    detector.known_face_encodings = []
                    detector.known_face_names = []
                    detector.save_face_database()
                    print("Face database reset!")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nSession complete. Processed {frame_count} frames.")
        print(f"Face database contains {len(detector.known_face_names)} known faces.")

if __name__ == "__main__":
    main()