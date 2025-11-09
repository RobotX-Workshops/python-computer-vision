"""
Camera Capture Face Recognition - Module 2 Exercise 4
Capture photos from camera and use them for face recognition
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import face_recognition
import numpy as np


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the camera capture demo"""
    parser = argparse.ArgumentParser(
        description="Interactive face recognition demo with configurable video sources"
    )
    parser.add_argument(
        "--video-source",
        default=None,
        help="Camera index (e.g. 0) or network stream URL (e.g. udp://host.docker.internal:5000)"
    )
    return parser.parse_args()


def resolve_video_source(cli_value: Optional[str]) -> str:
    """Resolve the active video source using CLI and environment fallbacks"""
    env_value = os.getenv("VIDEO_SOURCE")
    if cli_value:
        return cli_value
    if env_value:
        return env_value
    return "0"

class CameraFaceRecognition:
    """
    Capture photos from camera and perform face recognition
    """

    def __init__(self, video_source: Optional[str] = None):
        self.known_face_encodings = []
        self.known_face_names = []
        self.captured_images = []
        self.video_source = video_source or os.getenv("VIDEO_SOURCE", "0")

    def set_video_source(self, source: str) -> None:
        """Update the active video source used by capture methods"""
        self.video_source = source
        print(f"Video source updated to: {self.video_source}")

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Create a cv2.VideoCapture for the configured source"""
        source = self.video_source
        if source is None:
            source = "0"
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Cannot access video source {source}")
            return None
        return cap

    def capture_photo_from_camera(self, photo_name="captured_photo"):
        """
        Capture a photo from the camera
        
        Args:
            photo_name: Base name for the saved photo
            
        Returns:
            filename: Path to the saved photo, or None if failed
        """
        cap = self._open_capture()

        if cap is None:
            return None
        
        print("=== Camera Capture Mode ===")
        print("Position yourself in front of the camera")
        print("Press SPACE to capture photo, 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read from camera")
                break
            
            # Add instructions on the frame
            cv2.putText(frame, "Press SPACE to capture, 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Make sure your face is clearly visible", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Camera - Press SPACE to capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space key
                # Capture the photo
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = DATA_DIR / f"{photo_name}_{timestamp}.jpg"

                cv2.imwrite(str(filename), frame)
                print(f"Photo captured and saved as: {filename}")
                
                cap.release()
                cv2.destroyAllWindows()
                return filename
                
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return None
    
    def analyze_captured_photo(self, image_path):
        """
        Analyze a captured photo for faces
        
        Args:
            image_path: Path to the captured image
            
        Returns:
            face_info: Dictionary with face analysis results
        """
        print(f"\n=== Analyzing {image_path} ===")
        
        # Load image
        image_path = Path(image_path)
        if not image_path.is_absolute():
            image_path = DATA_DIR / image_path

        image = cv2.imread(str(image_path))
        if image is None:
            print("Error: Could not load image")
            return None
        
        # Convert to RGB for face_recognition library
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find faces
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        print(f"Found {len(face_locations)} face(s) in the image")
        
        # Create annotated image
        annotated_image = image.copy()
        
        face_info = {
            'image_path': str(image_path),
            'face_count': len(face_locations),
            'face_locations': face_locations,
            'face_encodings': face_encodings,
            'annotated_image': annotated_image
        }
        
        # Draw rectangles around faces
        for i, (top, right, bottom, left) in enumerate(face_locations):
            # Draw rectangle
            cv2.rectangle(annotated_image, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Add face number
            cv2.putText(annotated_image, f'Face {i+1}', (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Print face dimensions
            width = right - left
            height = bottom - top
            print(f"  Face {i+1}: {width}x{height} pixels at ({left}, {top})")
        
        return face_info
    
    def add_face_to_database(self, face_info, person_name):
        """
        Add a face from the captured photo to the recognition database
        
        Args:
            face_info: Face information from analyze_captured_photo
            person_name: Name to associate with the face
            
        Returns:
            success: Boolean indicating success
        """
        if face_info is None or len(face_info['face_encodings']) == 0:
            print("No faces found to add to database")
            return False
        
        if len(face_info['face_encodings']) > 1:
            print(f"Multiple faces found ({len(face_info['face_encodings'])}). Using the first one.")
            print("For better results, capture photos with only one person.")
        
        # Use the first face encoding
        face_encoding = face_info['face_encodings'][0]
        
        # Add to database
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(person_name)
        
        print(f"✅ Added {person_name} to face recognition database")
        print(f"Database now contains {len(self.known_face_names)} known faces")
        
        return True
    
    def recognize_faces_in_photo(self, face_info):
        """
        Try to recognize faces in a photo using the current database
        
        Args:
            face_info: Face information from analyze_captured_photo
            
        Returns:
            recognition_results: List of recognition results
        """
        if len(self.known_face_encodings) == 0:
            print("No faces in database yet. Capture and add some faces first.")
            return []
        
        recognition_results = []
        
        for i, face_encoding in enumerate(face_info['face_encodings']):
            # Compare with known faces
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(distances)
            confidence = 1 - distances[best_match_index]
            
            # Determine if it's a match (tolerance = 0.6)
            if distances[best_match_index] < 0.6:
                name = self.known_face_names[best_match_index]
                match_status = "MATCH"
            else:
                name = "Unknown"
                match_status = "NO MATCH"
            
            result = {
                'face_number': i + 1,
                'name': name,
                'confidence': confidence,
                'distance': distances[best_match_index],
                'match_status': match_status
            }
            
            recognition_results.append(result)
            
            print(f"Face {i+1}: {match_status} - {name} (confidence: {confidence:.2f}, distance: {distances[best_match_index]:.2f})")
        
        return recognition_results
    
    def create_annotated_result(self, face_info, recognition_results):
        """
        Create an annotated image showing recognition results
        
        Args:
            face_info: Face information
            recognition_results: Recognition results
            
        Returns:
            annotated_image: Image with recognition annotations
        """
        annotated_image = face_info['annotated_image'].copy()
        
        for i, (top, right, bottom, left) in enumerate(face_info['face_locations']):
            if i < len(recognition_results):
                result = recognition_results[i]
                
                # Choose color based on match status
                if result['match_status'] == "MATCH":
                    color = (0, 255, 0)  # Green for match
                else:
                    color = (0, 0, 255)  # Red for no match
                
                # Redraw rectangle with appropriate color
                cv2.rectangle(annotated_image, (left, top), (right, bottom), color, 3)
                
                # Add recognition label
                label = f"{result['name']} ({result['confidence']:.2f})"
                cv2.putText(annotated_image, label, (left, bottom + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return annotated_image
    
    def interactive_session(self):
        """
        Run an interactive face recognition session
        """
        print("=== Interactive Face Recognition Session ===")
        
        while True:
            print("\n--- Menu ---")
            print("1. Capture photo and analyze faces")
            print("2. Add person to database (from last photo)")
            print("3. Recognize faces (in last photo)")
            print("4. Show database status")
            print("5. Capture and test recognition")
            print("6. Change video source (current: {0})".format(self.video_source))
            print("7. Exit")
            
            try:
                choice = input("\nEnter your choice (1-7): ").strip()
                
                if choice == '1':
                    # Capture and analyze
                    filename = self.capture_photo_from_camera("person")
                    if filename:
                        self.current_face_info = self.analyze_captured_photo(filename)
                        if self.current_face_info and self.current_face_info['face_count'] > 0:
                            # Show the result
                            cv2.imshow('Captured Photo with Faces', self.current_face_info['annotated_image'])
                            print("Press any key to continue...")
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                        else:
                            print("No faces detected in the captured photo.")
                
                elif choice == '2':
                    # Add person to database
                    if hasattr(self, 'current_face_info') and self.current_face_info:
                        name = input("Enter the person's name: ").strip()
                        if name:
                            self.add_face_to_database(self.current_face_info, name)
                    else:
                        print("No photo analyzed yet. Choose option 1 first.")
                
                elif choice == '3':
                    # Recognize faces
                    if hasattr(self, 'current_face_info') and self.current_face_info:
                        results = self.recognize_faces_in_photo(self.current_face_info)
                        if results:
                            # Show annotated result
                            annotated = self.create_annotated_result(self.current_face_info, results)
                            cv2.imshow('Face Recognition Results', annotated)
                            print("Press any key to continue...")
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                    else:
                        print("No photo analyzed yet. Choose option 1 first.")
                
                elif choice == '4':
                    # Show database status
                    print("\n--- Database Status ---")
                    print("Known faces: {0}".format(len(self.known_face_names)))
                    if self.known_face_names:
                        for i, name in enumerate(self.known_face_names):
                            print(f"  {i+1}. {name}")
                    else:
                        print("  (No faces in database)")
                
                elif choice == '5':
                    # Capture and test recognition in one go
                    filename = self.capture_photo_from_camera("test")
                    if filename:
                        face_info = self.analyze_captured_photo(filename)
                        if face_info and face_info['face_count'] > 0:
                            if len(self.known_face_encodings) > 0:
                                results = self.recognize_faces_in_photo(face_info)
                                annotated = self.create_annotated_result(face_info, results)
                                cv2.imshow('Recognition Test Results', annotated)
                                print("Press any key to continue...")
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
                            else:
                                print("No faces in database. Add some faces first (option 2).")
                                cv2.imshow('Captured Photo', face_info['annotated_image'])
                                print("Press any key to continue...")
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
                
                elif choice == '6':
                    new_source = input("Enter camera index or stream URL: ").strip()
                    if new_source:
                        self.set_video_source(new_source)
                    else:
                        print("Video source unchanged.")

                elif choice == '7':
                    break
                
                else:
                    print("Invalid choice. Please enter 1-7.")
            
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break
        
        print("Session ended. Thanks for using face recognition!")

def main():
    """
    Main function
    """
    print("=== Camera-based Face Recognition Demo ===")
    print("This demo captures photos from your camera and uses them for face recognition.")

    args = parse_args()
    video_source = resolve_video_source(args.video_source)
    print(f"Video source: {video_source}")

    # Initialize the face recognition system
    face_rec = CameraFaceRecognition(video_source)

    # Run interactive session
    face_rec.interactive_session()

if __name__ == "__main__":
    main()