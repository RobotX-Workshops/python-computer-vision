"""
Real-time Canny Edge Detection - Module 1 Exercise 3
Apply Canny edge detection to webcam feed in real-time
"""

import cv2
import numpy as np

class RealTimeCanny:
    """
    Real-time Canny edge detection from webcam feed
    """
    
    def __init__(self):
        self.low_threshold = 50
        self.high_threshold = 150
        self.blur_kernel = 5
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open webcam")
            return
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
    
    def create_trackbars(self):
        """
        Create trackbars for real-time parameter adjustment
        """
        cv2.namedWindow('Controls')
        cv2.createTrackbar('Low Threshold', 'Controls', self.low_threshold, 255, self.update_low_threshold)
        cv2.createTrackbar('High Threshold', 'Controls', self.high_threshold, 255, self.update_high_threshold)
        cv2.createTrackbar('Blur Kernel', 'Controls', self.blur_kernel, 15, self.update_blur_kernel)
        
    def update_low_threshold(self, val):
        """Update low threshold value"""
        self.low_threshold = val
        
    def update_high_threshold(self, val):
        """Update high threshold value"""
        self.high_threshold = val
        
    def update_blur_kernel(self, val):
        """Update blur kernel size (ensure it's odd)"""
        if val < 1:
            val = 1
        if val % 2 == 0:
            val += 1
        self.blur_kernel = val
    
    def process_frame(self, frame):
        """
        Process a single frame with Canny edge detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
        
        # Convert edges back to 3-channel for display
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return gray, blurred, edges, edges_colored
    
    def add_text_overlay(self, frame, text_lines):
        """
        Add text overlay to frame
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0)  # Green
        thickness = 2
        
        y_offset = 30
        for line in text_lines:
            cv2.putText(frame, line, (10, y_offset), font, font_scale, color, thickness)
            y_offset += 25
    
    def run(self):
        """
        Main loop for real-time edge detection
        """
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            print("Webcam not available. Please check your camera connection.")
            return
        
        # Create control trackbars
        self.create_trackbars()
        
        print("=== Real-time Canny Edge Detection ===")
        print("Controls:")
        print("- Use trackbars to adjust parameters in real-time")
        print("- Press 'q' to quit")
        print("- Press 's' to save current frame")
        print("- Press 'r' to reset parameters")
        print("- Press 'h' to show/hide help text")
        
        show_help = True
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Cannot read frame from webcam")
                    break
                
                frame_count += 1
                
                # Process frame
                gray, blurred, edges, edges_colored = self.process_frame(frame)
                
                # Create combined display
                # Top row: Original and Grayscale
                top_row = np.hstack([frame, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)])
                
                # Bottom row: Blurred and Edges
                bottom_row = np.hstack([cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR), edges_colored])
                
                # Combine both rows
                combined = np.vstack([top_row, bottom_row])
                
                # Add labels
                label_font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(combined, 'Original', (10, 25), label_font, 0.7, (255, 255, 255), 2)
                cv2.putText(combined, 'Grayscale', (650, 25), label_font, 0.7, (255, 255, 255), 2)
                cv2.putText(combined, 'Blurred', (10, 505), label_font, 0.7, (255, 255, 255), 2)
                cv2.putText(combined, 'Edges', (650, 505), label_font, 0.7, (255, 255, 255), 2)
                
                # Add parameter information
                if show_help:
                    param_text = [
                        f"Low Threshold: {self.low_threshold}",
                        f"High Threshold: {self.high_threshold}",
                        f"Blur Kernel: {self.blur_kernel}x{self.blur_kernel}",
                        f"Frame: {frame_count}",
                        "Press 'h' to hide help"
                    ]
                    
                    # Add semi-transparent background for text
                    overlay = combined.copy()
                    cv2.rectangle(overlay, (10, 50), (300, 200), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, combined, 0.3, 0, combined)
                    
                    self.add_text_overlay(combined, param_text)
                
                # Display the result
                cv2.imshow('Real-time Canny Edge Detection', combined)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f'canny_frame_{frame_count}.png'
                    cv2.imwrite(filename, combined)
                    print(f"Saved frame as {filename}")
                elif key == ord('r'):
                    # Reset parameters
                    self.low_threshold = 50
                    self.high_threshold = 150
                    self.blur_kernel = 5
                    cv2.setTrackbarPos('Low Threshold', 'Controls', self.low_threshold)
                    cv2.setTrackbarPos('High Threshold', 'Controls', self.high_threshold)
                    cv2.setTrackbarPos('Blur Kernel', 'Controls', self.blur_kernel)
                    print("Parameters reset to defaults")
                elif key == ord('h'):
                    # Toggle help display
                    show_help = not show_help
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            # Clean up
            self.cap.release()
            cv2.destroyAllWindows()
            print("Real-time edge detection stopped")

def test_with_static_image():
    """
    Test the edge detection with a static image if webcam is not available
    """
    print("=== Testing with Static Images ===")
    
    # Create a test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some shapes
    cv2.rectangle(test_img, (100, 100), (300, 200), (255, 255, 255), -1)
    cv2.circle(test_img, (450, 150), 80, (128, 128, 128), -1)
    cv2.rectangle(test_img, (200, 300), (400, 400), (200, 200, 200), 5)
    
    # Add some noise
    noise = np.random.normal(0, 20, test_img.shape)
    test_img = np.clip(test_img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    
    # Create Canny processor
    processor = RealTimeCanny()
    
    # Process the test image with different parameters
    parameters = [
        (50, 150, 5, "Default"),
        (30, 100, 5, "More Sensitive"),
        (100, 200, 5, "Less Sensitive"),
        (50, 150, 9, "More Blur")
    ]
    
    results = []
    
    for low, high, blur, label in parameters:
        processor.low_threshold = low
        processor.high_threshold = high
        processor.blur_kernel = blur
        
        gray, blurred, edges, edges_colored = processor.process_frame(test_img)
        
        # Add label to the image
        labeled_edges = edges_colored.copy()
        cv2.putText(labeled_edges, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(labeled_edges, f"L:{low} H:{high} B:{blur}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        results.append(labeled_edges)
    
    # Display results
    top_row = np.hstack([test_img] + [cv2.cvtColor(cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)])
    bottom_row = np.hstack(results[:2])
    third_row = np.hstack(results[2:])
    
    # Add labels for top row
    cv2.putText(top_row, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(top_row, 'Grayscale', (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    combined = np.vstack([top_row, bottom_row, third_row])
    
    cv2.imshow('Static Image Canny Test', combined)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """
    Main function
    """
    print("=== Real-time Canny Edge Detection Demo ===")
    
    # Try to use webcam first
    test_cap = cv2.VideoCapture(0)
    if test_cap.isOpened():
        test_cap.release()
        print("Webcam detected. Starting real-time demo...")
        
        canny_detector = RealTimeCanny()
        canny_detector.run()
    else:
        print("No webcam detected. Running static image test...")
        test_with_static_image()

if __name__ == "__main__":
    main()