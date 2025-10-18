"""
Simple Camera Face Recognition Demo - Module 2 Exercise 4
Capture a photo and analyze faces - simplified version
"""

import argparse
import os
from datetime import datetime
from typing import Optional

import cv2
import face_recognition


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the simple demo"""
    parser = argparse.ArgumentParser(
        description="Capture a photo for face analysis using a local camera or stream"
    )
    parser.add_argument(
        "--video-source",
        default=None,
        help="Camera index (e.g. 0) or stream URL (e.g. udp://host.docker.internal:5000)"
    )
    return parser.parse_args()


def resolve_video_source(cli_value: Optional[str]) -> str:
    """Resolve the video source from CLI or the VIDEO_SOURCE environment variable"""
    env_value = os.getenv("VIDEO_SOURCE")
    if cli_value:
        return cli_value
    if env_value:
        return env_value
    return "0"


def open_capture(video_source: str) -> Optional[cv2.VideoCapture]:
    """Create a VideoCapture using either a numeric index or URL"""
    if video_source.isdigit():
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"❌ Error: Cannot access video source {video_source}")
        return None
    return cap


def capture_photo(video_source: str):
    """
    Capture a single photo from camera
    """
    print("=== Capturing Photo from Camera ===")
    print("Make sure you're positioned well in front of the camera")
    print("Press SPACE when ready, or 'q' to quit")
    
    cap = open_capture(video_source)

    if cap is None:
        return None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Add instructions
        cv2.putText(frame, "Press SPACE to capture photo", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Camera Preview', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space to capture
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_photo_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✅ Photo saved as: {filename}")
            
            cap.release()
            cv2.destroyAllWindows()
            return filename
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return None

def analyze_faces_in_photo(image_path):
    """
    Analyze faces in the captured photo
    """
    print(f"\n=== Analyzing faces in {image_path} ===")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Error: Could not load image")
        return None
    
    # Convert to RGB for face_recognition
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    print("🔍 Detecting faces...")
    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    
    print(f"📊 Found {len(face_locations)} face(s)")
    
    if len(face_locations) == 0:
        print("😔 No faces detected. Try capturing another photo with better lighting.")
        return None
    
    # Get face encodings
    print("🧠 Creating face encodings...")
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    # Create annotated image
    annotated_image = image.copy()
    
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Draw rectangle around face
        cv2.rectangle(annotated_image, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Add face label
        cv2.putText(annotated_image, f'Person {i+1}', (left, top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Calculate face size
        width = right - left
        height = bottom - top
        print(f"   Face {i+1}: {width}x{height} pixels")
    
    # Show the annotated result
    cv2.imshow('Detected Faces', annotated_image)
    print("👀 Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return {
        'image_path': image_path,
        'original_image': image,
        'face_locations': face_locations,
        'face_encodings': face_encodings,
        'annotated_image': annotated_image
    }

def demonstrate_face_comparison(face_data):
    """
    Demonstrate face comparison if multiple faces are found
    """
    if len(face_data['face_encodings']) < 2:
        print("ℹ️  Only one face found - skipping comparison demo")
        return
    
    print("\n=== Face Comparison Demo ===")
    print(f"Comparing {len(face_data['face_encodings'])} faces found in the photo")
    
    # Compare each pair of faces
    for i in range(len(face_data['face_encodings'])):
        for j in range(i + 1, len(face_data['face_encodings'])):
            distance = face_recognition.face_distance(
                [face_data['face_encodings'][i]], 
                face_data['face_encodings'][j]
            )[0]
            
            # Determine if it's likely the same person
            if distance < 0.6:
                similarity = "SAME PERSON"
                confidence = f"{(1-distance)*100:.1f}%"
            else:
                similarity = "DIFFERENT PEOPLE"
                confidence = f"{(1-distance)*100:.1f}%"
            
            print(f"   Face {i+1} vs Face {j+1}: {similarity} (confidence: {confidence}, distance: {distance:.3f})")

def show_face_recognition_info():
    """
    Show information about face recognition
    """
    print("\n=== Face Recognition Information ===")
    print("📚 What just happened:")
    print("   1. 📷 Captured photo from your camera")
    print("   2. 🔍 Detected faces using deep learning")
    print("   3. 🧠 Created 128-dimensional face encodings")
    print("   4. 📏 Measured distances between face encodings")
    
    print("\n🎯 Key Concepts:")
    print("   • Face Detection: Finding faces in images")
    print("   • Face Encoding: Converting faces to numerical vectors")
    print("   • Face Distance: Measuring similarity (< 0.6 = same person)")
    print("   • Recognition: Identifying specific individuals")
    
    print("\n⚙️  Parameters:")
    print("   • Detection Model: 'hog' (fast) used here")
    print("   • Alternative: 'cnn' (more accurate, slower)")
    print("   • Distance Threshold: 0.6 (adjustable)")
    
    print("\n🔧 Next Steps:")
    print("   • Try the realtime_face_detection.py for live recognition")
    print("   • Build a database of known faces")
    print("   • Experiment with different lighting conditions")

def main():
    """
    Main demo function
    """
    print("=== Simple Camera Face Recognition Demo ===")
    print("This demo will:")
    print("1. 📷 Capture a photo from your camera")
    print("2. 🔍 Detect any faces in the photo")
    print("3. 🧠 Analyze the faces using face recognition")
    print("4. 📊 Show you the results")

    args = parse_args()
    video_source = resolve_video_source(args.video_source)
    print(f"Video source: {video_source}")
    
    # Step 1: Capture photo
    photo_path = capture_photo(video_source)
    if not photo_path:
        print("❌ No photo captured. Exiting.")
        return
    
    # Step 2: Analyze faces
    face_data = analyze_faces_in_photo(photo_path)
    if not face_data:
        print("❌ No faces found. Try again with better lighting.")
        return
    
    # Step 3: Demonstrate face comparison (if multiple faces)
    demonstrate_face_comparison(face_data)
    
    # Step 4: Show educational information
    show_face_recognition_info()
    
    print("\n✅ Demo complete!")
    print(f"📁 Your photo is saved as: {photo_path}")
    print("🔄 Run this script again to capture more photos!")

if __name__ == "__main__":
    main()