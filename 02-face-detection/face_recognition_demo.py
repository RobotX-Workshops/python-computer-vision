"""
Face Recognition Library Demo - Module 2 Exercise 3 (Simplified)
Demonstrate face_recognition library capabilities without requiring real photos
"""

import cv2
import face_recognition

def demonstrate_face_detection_methods():
    """
    Compare OpenCV Haar cascades with face_recognition library using webcam or sample
    """
    print("=== Face Detection Methods Comparison ===")
    
    # Initialize OpenCV face detector
    try:
        # Try using cv2.data first (newer OpenCV versions)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' # type: ignore
    except AttributeError:
        # Fallback for older OpenCV versions
        cascade_path = 'haarcascade_frontalface_default.xml'
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("Warning: Could not load face cascade classifier")
        print("Make sure OpenCV is properly installed with cascade files")
    
    print("\nMethod 1: OpenCV Haar Cascades")
    print("- Fast detection")
    print("- Good for real-time applications")
    print("- Can have false positives")
    print("- No face recognition capabilities")
    
    print("\nMethod 2: face_recognition Library")
    print("- More accurate detection")
    print("- Can identify specific individuals")
    print("- Deep learning based")
    print("- Slower but more reliable")
    
    return face_cascade

def demonstrate_face_encoding_concepts():
    """
    Explain face encoding concepts
    """
    print("\n=== Face Recognition Concepts ===")
    
    print("Face Encoding Process:")
    print("1. Face Detection: Locate faces in image")
    print("2. Face Alignment: Normalize face orientation")
    print("3. Feature Extraction: Create 128-dimensional vector")
    print("4. Face Comparison: Compare vectors using distance metrics")
    
    print("\nKey Parameters:")
    print("- Detection Model: 'hog' (fast) vs 'cnn' (accurate)")
    print("- Tolerance: Lower = stricter matching (0.4-0.6 typical)")
    print("- Face Locations: (top, right, bottom, left) coordinates")
    
    # Demonstrate distance calculation concept
    print("\nDistance Calculation:")
    print("- Same person: distance < 0.6")
    print("- Different people: distance > 0.6")
    print("- Identical images: distance ≈ 0.0")

def create_interactive_demo():
    """
    Create an interactive demo for face recognition concepts
    """
    print("\n=== Interactive Face Recognition Demo ===")
    print("This demo shows how face recognition would work with real faces.")
    
    # Simulate face recognition process
    print("\nStep 1: Loading known faces...")
    print("  [Simulated] Loading John.jpg -> Creating face encoding")
    print("  [Simulated] Loading Jane.jpg -> Creating face encoding")
    print("  [Simulated] Database now contains 2 known faces")
    
    print("\nStep 2: Processing test image...")
    print("  [Simulated] Found 2 faces in test image")
    print("  [Simulated] Face 1: Distance to John = 0.3 (MATCH: John)")
    print("  [Simulated] Face 2: Distance to Jane = 0.8 (No match: Unknown)")
    
    print("\nStep 3: Results...")
    print("  Face 1: Identified as John (confidence: 70%)")
    print("  Face 2: Unknown person")
    
    # Show tolerance effects
    print("\n=== Tolerance Effects ===")
    tolerances = [0.4, 0.5, 0.6, 0.7, 0.8]
    test_distance = 0.65
    
    print(f"Example: Face distance = {test_distance}")
    for tolerance in tolerances:
        if test_distance < tolerance:
            result = "MATCH"
        else:
            result = "NO MATCH"
        print(f"  Tolerance {tolerance}: {result}")

def webcam_detection_demo():
    """
    Real-time face detection demo using webcam
    """
    print("\n=== Real-time Face Detection Demo ===")
    print("Press 'q' to quit, 'h' for help")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam available. Skipping real-time demo.")
        return
    
    frame_count = 0
    face_locations = []  # Initialize face_locations
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert BGR to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces (process every 3rd frame for performance)
            if frame_count % 3 == 0:
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            
            # Draw detected faces
            for (top, right, bottom, left) in face_locations:
                # Draw rectangle
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Add label
                cv2.putText(frame, 'Face Detected', (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add info text
            info_text = f"Faces: {len(face_locations)} | Frame: {frame_count}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Face Detection Demo', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                print("\nControls:")
                print("  'q' - Quit")
                print("  'h' - Show this help")
    
    except KeyboardInterrupt:
        print("Demo interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Demo completed. Processed {frame_count} frames.")

def show_face_recognition_workflow():
    """
    Show the complete face recognition workflow
    """
    print("\n=== Face Recognition Workflow ===")
    
    workflow_steps = [
        ("1. Image Input", "Load image or capture from webcam"),
        ("2. Face Detection", "Locate all faces in the image"),
        ("3. Face Encoding", "Convert each face to 128-D vector"),
        ("4. Database Comparison", "Compare with known face encodings"),
        ("5. Distance Calculation", "Compute similarity scores"),
        ("6. Recognition Decision", "Match or 'Unknown' based on tolerance"),
        ("7. Results Display", "Show identified faces with confidence")
    ]
    
    for step, description in workflow_steps:
        print(f"{step}: {description}")
    
    print("\nCode Example Structure:")
    print("""
# Basic face recognition code structure:
import face_recognition

# 1. Load and encode known faces
known_image = face_recognition.load_image_file("person.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# 2. Process test image  
test_image = face_recognition.load_image_file("test.jpg")
test_encodings = face_recognition.face_encodings(test_image)

# 3. Compare faces
for encoding in test_encodings:
    distances = face_recognition.face_distance([known_encoding], encoding)
    if distances[0] < 0.6:
        print("Match found!")
    else:
        print("No match")
    """)

def main():
    """
    Main demonstration function
    """
    print("=== Face Recognition Library Demo ===")
    print("This demo explains face recognition concepts and shows basic detection.")
    
    # Demonstrate concepts
    face_cascade = demonstrate_face_detection_methods()
    demonstrate_face_encoding_concepts()
    create_interactive_demo()
    show_face_recognition_workflow()
    
    # Offer real-time demo
    print("\n=== Real-time Demo Options ===")
    print("1. Webcam face detection (if camera available)")
    print("2. Skip to summary")
    
    try:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice == "1":
            webcam_detection_demo()
        else:
            print("Skipping webcam demo.")
    except (EOFError, KeyboardInterrupt):
        print("\nSkipping interactive portions.")
    
    print("\n=== Demo Summary ===")
    print("✅ Face Detection Methods Compared")
    print("✅ Face Encoding Concepts Explained")  
    print("✅ Recognition Workflow Demonstrated")
    print("✅ Tolerance Effects Illustrated")
    
    print("\nTo use face recognition with real photos:")
    print("1. Install: pip install face_recognition")
    print("2. Add face photos to this directory")
    print("3. Run: python realtime_face_detection.py")
    print("4. Use 'a' key to add faces to database")
    
    print("\nNext Steps:")
    print("- Try the real-time face detection script")
    print("- Experiment with different tolerance values")
    print("- Build your own face recognition database")
    print("- Explore face landmark detection")

if __name__ == "__main__":
    main()