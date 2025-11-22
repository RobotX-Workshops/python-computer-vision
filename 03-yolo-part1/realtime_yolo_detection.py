"""
Real-time YOLO Detection - Module 3 Exercise 2
Implement real-time object detection on video streams
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
import os


class RealtimeYOLODetector:
    """
    Real-time object detection using YOLO models
    """
    
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5, iou_threshold=0.5):
        """
        Initialize real-time YOLO detector
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detection
            iou_threshold: IoU threshold for NMS
        """
        try:
            self.model = YOLO(model_path)
            print(f"YOLO model loaded: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Downloading YOLOv8 nano model...")
            self.model = YOLO('yolov8n.pt')
        
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Performance tracking
        self.fps_history = []
        self.max_fps_history = 30
        
        # COCO class names
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
    
    def update_fps(self, fps):
        """
        Update FPS history for smoothing
        
        Args:
            fps: Current FPS value
        """
        self.fps_history.append(fps)
        if len(self.fps_history) > self.max_fps_history:
            self.fps_history.pop(0)
    
    def get_average_fps(self):
        """
        Get average FPS from history
        
        Returns:
            Average FPS
        """
        if len(self.fps_history) == 0:
            return 0
        return sum(self.fps_history) / len(self.fps_history)
    
    def draw_info_panel(self, frame, detections, fps):
        """
        Draw information panel on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            fps: Current FPS
        
        Returns:
            Frame with info panel
        """
        # Semi-transparent overlay
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw background panel
        panel_height = 120
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # FPS info
        fps_text = f"FPS: {fps:.1f} (Avg: {self.get_average_fps():.1f})"
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Detection count
        det_text = f"Detections: {len(detections)}"
        cv2.putText(frame, det_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Confidence threshold
        conf_text = f"Confidence: {self.confidence_threshold:.2f} | IoU: {self.iou_threshold:.2f}"
        cv2.putText(frame, conf_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Controls hint
        controls_text = "Controls: q=quit, c/v=conf, n/m=iou, p=pause, s=save"
        cv2.putText(frame, controls_text, (10, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run_detection(self, video_source=0):
        """
        Run real-time object detection
        
        Args:
            video_source: Video source (0 for webcam, or path to video file)
        """
        print("\n=== Real-time YOLO Detection ===")
        print(f"Video source: {video_source}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"IoU threshold: {self.iou_threshold}")
        print("\nControls:")
        print("  q - Quit")
        print("  c - Increase confidence threshold")
        print("  v - Decrease confidence threshold")
        print("  n - Increase IoU threshold")
        print("  m - Decrease IoU threshold")
        print("  p - Pause/Resume")
        print("  s - Save current frame")
        
        # Open video source
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\nVideo resolution: {width}x{height}")
        
        paused = False
        frame_count = 0
        saved_count = 0
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("End of video or cannot read frame")
                        break
                    
                    frame_count += 1
                    start_time = time.time()
                    
                    # Run YOLO detection
                    results = self.model(frame, 
                                       conf=self.confidence_threshold,
                                       iou=self.iou_threshold,
                                       verbose=False)
                    
                    # Get annotated frame
                    annotated_frame = results[0].plot()
                    
                    # Calculate FPS
                    fps = 1.0 / (time.time() - start_time)
                    self.update_fps(fps)
                    
                    # Get detection info
                    detections = []
                    if results[0].boxes is not None:
                        boxes = results[0].boxes
                        for i in range(len(boxes)):
                            class_id = int(boxes.cls[i].cpu().numpy())
                            confidence = float(boxes.conf[i].cpu().numpy())
                            detections.append({
                                'class_id': class_id,
                                'class_name': self.class_names[class_id],
                                'confidence': confidence
                            })
                    
                    # Draw info panel
                    display_frame = self.draw_info_panel(annotated_frame, detections, fps)
                else:
                    # Display paused frame
                    display_frame = annotated_frame.copy()
                    cv2.putText(display_frame, "PAUSED", (width//2 - 100, height//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                
                # Display frame
                cv2.imshow('YOLO Real-time Detection', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                
                elif key == ord('c'):
                    self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                    print(f"Confidence threshold: {self.confidence_threshold:.2f}")
                
                elif key == ord('v'):
                    self.confidence_threshold = max(0.05, self.confidence_threshold - 0.05)
                    print(f"Confidence threshold: {self.confidence_threshold:.2f}")
                
                elif key == ord('n'):
                    self.iou_threshold = min(0.95, self.iou_threshold + 0.05)
                    print(f"IoU threshold: {self.iou_threshold:.2f}")
                
                elif key == ord('m'):
                    self.iou_threshold = max(0.05, self.iou_threshold - 0.05)
                    print(f"IoU threshold: {self.iou_threshold:.2f}")
                
                elif key == ord('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                
                elif key == ord('s'):
                    filename = f"yolo_capture_{saved_count:03d}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Saved: {filename}")
                    saved_count += 1
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Print statistics
            print("\n=== Session Statistics ===")
            print(f"Frames processed: {frame_count}")
            print(f"Average FPS: {self.get_average_fps():.1f}")
            print(f"Frames saved: {saved_count}")


def main():
    """
    Main function for real-time YOLO detection
    """
    parser = argparse.ArgumentParser(description='Real-time YOLO Object Detection')
    parser.add_argument('--video-source', type=str, default='0',
                       help='Video source (0 for webcam, or path/URL to video)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model path (default: yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold for NMS (default: 0.5)')
    
    args = parser.parse_args()
    
    # Handle video source
    video_source = args.video_source
    if video_source.isdigit():
        video_source = int(video_source)
    elif not os.path.exists(video_source) and not video_source.startswith(('http://', 'https://', 'rtsp://', 'udp://')):
        print(f"Warning: Video source '{video_source}' not found")
        print("Falling back to webcam (0)")
        video_source = 0
    
    # Check for environment variable
    env_video_source = os.environ.get('VIDEO_SOURCE')
    if env_video_source:
        if env_video_source.isdigit():
            video_source = int(env_video_source)
        else:
            video_source = env_video_source
        print(f"Using VIDEO_SOURCE from environment: {video_source}")
    
    # Initialize detector
    detector = RealtimeYOLODetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Run detection
    detector.run_detection(video_source) # type: ignore


if __name__ == "__main__":
    main()
