"""
Image Segmentation - Module 3 Bonus Exercise
Learn semantic and instance segmentation alongside object detection
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

class ImageSegmentationDemo:
    """
    Demonstrate different types of image segmentation
    """
    
    def __init__(self):
        """
        Initialize segmentation models
        """
        # Load YOLO segmentation model (YOLOv8 segment)
        try:
            self.yolo_seg = YOLO('yolov8n-seg.pt')  # Segmentation version
            print("YOLO segmentation model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO segmentation model: {e}")
            self.yolo_seg = None
        
        # COCO class names for reference
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
    
    def traditional_segmentation(self, image):
        """
        Demonstrate traditional segmentation methods
        
        Args:
            image: Input image
            
        Returns:
            results: Dictionary of segmentation results
        """
        print("Applying traditional segmentation methods...")
        
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 1. Thresholding
        _, thresh_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. K-means clustering for color segmentation
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = 5  # Number of clusters
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8 and reshape
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        kmeans_result = segmented_data.reshape(image.shape)
        
        # 3. Watershed segmentation
        watershed_result = self.watershed_segmentation(image)
        
        # 4. GrabCut (interactive foreground extraction)
        grabcut_result = self.grabcut_segmentation(image)
        
        return {
            'original': image,
            'gray': gray,
            'thresh_binary': thresh_binary,
            'thresh_otsu': thresh_otsu,
            'kmeans': kmeans_result,
            'watershed': watershed_result,
            'grabcut': grabcut_result
        }
    
    def watershed_segmentation(self, image):
        """
        Apply watershed segmentation
        
        Args:
            image: Input image
            
        Returns:
            watershed_result: Segmented image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        result = image.copy()
        result[markers == -1] = [255, 0, 0]  # Mark boundaries in red
        
        return result
    
    def grabcut_segmentation(self, image):
        """
        Apply GrabCut algorithm for foreground extraction
        
        Args:
            image: Input image
            
        Returns:
            grabcut_result: Foreground extracted image
        """
        height, width = image.shape[:2]
        
        # Define rectangle around the object (center 60% of image)
        margin_x = width // 5
        margin_y = height // 5
        rect = (margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y)
        
        # Initialize mask
        mask = np.zeros((height, width), np.uint8)
        
        # Initialize background and foreground models
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        # Modify mask: 0 and 2 are background, 1 and 3 are foreground
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Apply mask to image
        result = image * mask2[:, :, np.newaxis]
        
        return result
    
    def yolo_segmentation(self, image):
        """
        Apply YOLO instance segmentation
        
        Args:
            image: Input image
            
        Returns:
            results: YOLO segmentation results
            annotated_image: Image with segmentation masks
            masks: Individual segmentation masks
        """
        if self.yolo_seg is None:
            print("YOLO segmentation model not available")
            return None, image, []
        
        print("Applying YOLO instance segmentation...")
        
        # Run inference
        results = self.yolo_seg(image, conf=0.5)
        
        # Get annotated image with masks
        annotated_image = results[0].plot()
        
        # Extract individual masks
        masks = []
        if results[0].masks is not None:
            mask_data = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for i in range(len(mask_data)):
                mask_info = {
                    'mask': mask_data[i],
                    'bbox': boxes[i],
                    'class_id': class_ids[i],
                    'class_name': self.class_names[class_ids[i]],
                    'confidence': confidences[i]
                }
                masks.append(mask_info)
        
        return results[0], annotated_image, masks
    
    def create_colored_masks(self, image, masks):
        """
        Create colored overlay of segmentation masks
        
        Args:
            image: Original image
            masks: List of mask information
            
        Returns:
            colored_overlay: Image with colored masks
        """
        overlay = image.copy()
        
        # Generate random colors for each mask
        colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)
        
        for i, mask_info in enumerate(masks):
            mask = mask_info['mask']
            color = colors[i]
            
            # Resize mask to image dimensions if needed
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # Create colored mask
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0.5] = color
            
            # Blend with original image
            alpha = 0.4
            overlay = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
        
        return overlay
    
    def compare_segmentation_methods(self, image_path):
        """
        Compare different segmentation methods on the same image
        
        Args:
            image_path: Path to input image
        """
        print(f"Comparing segmentation methods on: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        # Apply traditional methods
        traditional_results = self.traditional_segmentation(image)
        
        # Apply YOLO segmentation
        yolo_results, yolo_annotated, yolo_masks = self.yolo_segmentation(image)
        
        # Create visualization
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Image Segmentation Comparison', fontsize=16)
        
        # Row 1: Traditional methods
        axes[0, 0].imshow(cv2.cvtColor(traditional_results['original'], cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(traditional_results['thresh_otsu'], cmap='gray')
        axes[0, 1].set_title('Otsu Threshold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(cv2.cvtColor(traditional_results['kmeans'], cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('K-means Clustering (k=5)')
        axes[0, 2].axis('off')
        
        # Row 2: More traditional methods
        axes[1, 0].imshow(cv2.cvtColor(traditional_results['watershed'], cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Watershed Segmentation')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cv2.cvtColor(traditional_results['grabcut'], cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('GrabCut Foreground')
        axes[1, 1].axis('off')
        
        # YOLO segmentation
        axes[1, 2].imshow(cv2.cvtColor(yolo_annotated, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f'YOLO Instance Segmentation\n({len(yolo_masks)} objects)')
        axes[1, 2].axis('off')
        
        # Row 3: Individual YOLO masks and colored overlay
        if len(yolo_masks) > 0:
            # Show first mask if available
            if len(yolo_masks) > 0:
                first_mask = yolo_masks[0]['mask']
                if first_mask.shape != image.shape[:2]:
                    first_mask = cv2.resize(first_mask, (image.shape[1], image.shape[0]))
                axes[2, 0].imshow(first_mask, cmap='gray')
                axes[2, 0].set_title(f'First Mask: {yolo_masks[0]["class_name"]}')
            else:
                axes[2, 0].text(0.5, 0.5, 'No masks found', ha='center', va='center')
                axes[2, 0].set_title('No Masks')
            axes[2, 0].axis('off')
            
            # Colored overlay
            colored_overlay = self.create_colored_masks(image, yolo_masks)
            axes[2, 1].imshow(cv2.cvtColor(colored_overlay, cv2.COLOR_BGR2RGB))
            axes[2, 1].set_title('Colored Mask Overlay')
            axes[2, 1].axis('off')
            
            # Statistics
            stats_text = f"Detected Objects:\n"
            class_counts = {}
            for mask in yolo_masks:
                class_name = mask['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            for class_name, count in class_counts.items():
                avg_conf = np.mean([m['confidence'] for m in yolo_masks if m['class_name'] == class_name])
                stats_text += f"{class_name}: {count} (conf: {avg_conf:.2f})\n"
            
            axes[2, 2].text(0.1, 0.9, stats_text, transform=axes[2, 2].transAxes, 
                           verticalalignment='top', fontfamily='monospace')
            axes[2, 2].set_title('Detection Statistics')
            axes[2, 2].axis('off')
        else:
            for j in range(3):
                axes[2, j].text(0.5, 0.5, 'No YOLO masks detected', ha='center', va='center')
                axes[2, j].set_title('No Results')
                axes[2, j].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print("\n=== Segmentation Results ===")
        print(f"Traditional methods applied successfully")
        print(f"YOLO detected {len(yolo_masks)} objects with segmentation masks")
        
        if yolo_masks:
            print("\nDetected objects:")
            for i, mask in enumerate(yolo_masks):
                print(f"  {i+1}. {mask['class_name']} (confidence: {mask['confidence']:.2f})")

def main():
    """
    Main demonstration function
    """
    print("=== Image Segmentation Demo ===")
    print("This demo compares traditional and modern segmentation methods")
    
    # Initialize segmentation demo
    seg_demo = ImageSegmentationDemo()
    
    # Look for test image
    test_image_path = 'yolo_test_scene.jpeg'
    if not os.path.exists(test_image_path):
        print(f"Test image {test_image_path} not found.")
        print("Please make sure you have a test image in the current directory.")
        return
    
    print(f"\nUsing test image: {test_image_path}")
    
    try:
        # Run segmentation comparison
        seg_demo.compare_segmentation_methods(test_image_path)
        
        print("\n=== Exercise Complete ===")
        print("\nKey Concepts Learned:")
        print("1. Traditional Segmentation:")
        print("   - Thresholding: Separates objects based on intensity")
        print("   - K-means: Groups pixels by color similarity")
        print("   - Watershed: Uses gradient information for boundaries")
        print("   - GrabCut: Interactive foreground/background separation")
        
        print("\n2. Modern Instance Segmentation:")
        print("   - YOLO-seg: Provides both detection and pixel-level masks")
        print("   - Each object gets individual mask and classification")
        print("   - Much more accurate for complex scenes")
        
        print("\n3. Applications:")
        print("   - Medical imaging: Tumor/organ segmentation")
        print("   - Autonomous vehicles: Road/obstacle segmentation")
        print("   - Photo editing: Automatic subject isolation")
        print("   - Robotics: Object manipulation and grasping")
        
    except Exception as e:
        print(f"Error during segmentation demo: {e}")
        print("Make sure you have installed: ultralytics opencv-python matplotlib numpy")

if __name__ == "__main__":
    main()