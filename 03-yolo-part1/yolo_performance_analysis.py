"""
YOLO Performance Analysis - Module 3 Exercise 3
Analyze detection performance and parameter effects
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import os
from collections import defaultdict


class YOLOPerformanceAnalyzer:
    """
    Analyze YOLO model performance with different parameters
    """
    
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize performance analyzer
        
        Args:
            model_path: Path to YOLO model weights
        """
        try:
            self.model = YOLO(model_path)
            print(f"YOLO model loaded: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Downloading YOLOv8 nano model...")
            self.model = YOLO('yolov8n.pt')
        
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
    
    def benchmark_inference_speed(self, image, num_runs=100):
        """
        Benchmark inference speed
        
        Args:
            image: Test image
            num_runs: Number of inference runs
        
        Returns:
            Dictionary with timing statistics
        """
        print(f"\n=== Benchmarking Inference Speed ({num_runs} runs) ===")
        
        times = []
        
        # Warmup
        for _ in range(5):
            _ = self.model(image, verbose=False)
        
        # Actual benchmark
        for i in range(num_runs):
            start_time = time.time()
            _ = self.model(image, verbose=False)
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{num_runs} runs")
        
        stats = {
            'mean': np.mean(times) * 1000,  # Convert to ms
            'std': np.std(times) * 1000,
            'min': np.min(times) * 1000,
            'max': np.max(times) * 1000,
            'fps': 1.0 / np.mean(times)
        }
        
        print(f"\nInference Time Statistics:")
        print(f"  Mean: {stats['mean']:.2f} ms ± {stats['std']:.2f} ms")
        print(f"  Min:  {stats['min']:.2f} ms")
        print(f"  Max:  {stats['max']:.2f} ms")
        print(f"  FPS:  {stats['fps']:.2f}")
        
        return stats, times
    
    def analyze_confidence_threshold_effect(self, image):
        """
        Analyze the effect of different confidence thresholds
        
        Args:
            image: Test image
        
        Returns:
            Dictionary with analysis results
        """
        print("\n=== Analyzing Confidence Threshold Effects ===")
        
        thresholds = np.arange(0.1, 1.0, 0.1)
        results = {
            'thresholds': thresholds,
            'detection_counts': [],
            'avg_confidences': []
        }
        
        for threshold in thresholds:
            detections = self.model(image, conf=threshold, verbose=False)
            
            if detections[0].boxes is not None:
                num_detections = len(detections[0].boxes)
                confidences = detections[0].boxes.conf.cpu().numpy()
                avg_conf = np.mean(confidences) if len(confidences) > 0 else 0
            else:
                num_detections = 0
                avg_conf = 0
            
            results['detection_counts'].append(num_detections)
            results['avg_confidences'].append(avg_conf)
            
            print(f"  Conf ≥ {threshold:.1f}: {num_detections} detections (avg conf: {avg_conf:.3f})")
        
        return results
    
    def analyze_iou_threshold_effect(self, image):
        """
        Analyze the effect of different IoU thresholds for NMS
        
        Args:
            image: Test image
        
        Returns:
            Dictionary with analysis results
        """
        print("\n=== Analyzing IoU Threshold Effects (NMS) ===")
        
        iou_thresholds = np.arange(0.1, 1.0, 0.1)
        results = {
            'iou_thresholds': iou_thresholds,
            'detection_counts': []
        }
        
        for iou in iou_thresholds:
            detections = self.model(image, conf=0.25, iou=iou, verbose=False)
            
            num_detections = len(detections[0].boxes) if detections[0].boxes is not None else 0
            results['detection_counts'].append(num_detections)
            
            print(f"  IoU ≤ {iou:.1f}: {num_detections} detections")
        
        return results
    
    def analyze_image_size_impact(self, image):
        """
        Analyze impact of different image sizes on performance
        
        Args:
            image: Test image
        
        Returns:
            Dictionary with analysis results
        """
        print("\n=== Analyzing Image Size Impact ===")
        
        sizes = [320, 416, 512, 640, 800, 1024]
        results = {
            'sizes': sizes,
            'inference_times': [],
            'detection_counts': []
        }
        
        for size in sizes:
            # Resize image
            resized = cv2.resize(image, (size, size))
            
            # Benchmark
            times = []
            for _ in range(10):
                start = time.time()
                detections = self.model(resized, verbose=False)
                times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1000  # Convert to ms
            num_detections = len(detections[0].boxes) if detections[0].boxes is not None else 0
            
            results['inference_times'].append(avg_time)
            results['detection_counts'].append(num_detections)
            
            print(f"  Size {size}x{size}: {avg_time:.2f} ms, {num_detections} detections")
        
        return results
    
    def analyze_class_distribution(self, image):
        """
        Analyze detection class distribution
        
        Args:
            image: Test image
        
        Returns:
            Dictionary with class distribution
        """
        print("\n=== Analyzing Class Distribution ===")
        
        detections = self.model(image, conf=0.25, verbose=False)
        
        class_counts = defaultdict(int)
        class_confidences = defaultdict(list)
        
        if detections[0].boxes is not None:
            classes = detections[0].boxes.cls.cpu().numpy().astype(int)
            confidences = detections[0].boxes.conf.cpu().numpy()
            
            for cls, conf in zip(classes, confidences):
                class_name = self.class_names[cls]
                class_counts[class_name] += 1
                class_confidences[class_name].append(conf)
        
        print(f"\nDetected Classes:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            avg_conf = np.mean(class_confidences[class_name])
            print(f"  {class_name}: {count} (avg conf: {avg_conf:.3f})")
        
        return dict(class_counts), dict(class_confidences)
    
    def visualize_performance_analysis(self, image):
        """
        Create comprehensive performance visualization
        
        Args:
            image: Test image
        """
        print("\n=== Creating Performance Visualizations ===")
        
        # Run all analyses
        speed_stats, speed_times = self.benchmark_inference_speed(image, num_runs=50)
        conf_results = self.analyze_confidence_threshold_effect(image)
        iou_results = self.analyze_iou_threshold_effect(image)
        size_results = self.analyze_image_size_impact(image)
        
        # Create comprehensive plot
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Inference time distribution
        ax1 = plt.subplot(3, 3, 1)
        ax1.hist(speed_times, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(speed_times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(speed_times)*1000:.2f}ms')
        ax1.set_xlabel('Inference Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Inference Time Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Confidence threshold effect on detection count
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(conf_results['thresholds'], conf_results['detection_counts'], 
                marker='o', linewidth=2, markersize=8)
        ax2.set_xlabel('Confidence Threshold')
        ax2.set_ylabel('Number of Detections')
        ax2.set_title('Confidence Threshold vs Detection Count')
        ax2.grid(True, alpha=0.3)
        
        # 3. Confidence threshold effect on average confidence
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(conf_results['thresholds'], conf_results['avg_confidences'], 
                marker='s', linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('Confidence Threshold')
        ax3.set_ylabel('Average Confidence Score')
        ax3.set_title('Confidence Threshold vs Avg Confidence')
        ax3.grid(True, alpha=0.3)
        
        # 4. IoU threshold effect
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(iou_results['iou_thresholds'], iou_results['detection_counts'], 
                marker='^', linewidth=2, markersize=8, color='orange')
        ax4.set_xlabel('IoU Threshold (NMS)')
        ax4.set_ylabel('Number of Detections')
        ax4.set_title('IoU Threshold vs Detection Count\n(Higher IoU = Stricter NMS)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Image size vs inference time
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(size_results['sizes'], size_results['inference_times'], 
                marker='D', linewidth=2, markersize=8, color='red')
        ax5.set_xlabel('Image Size (pixels)')
        ax5.set_ylabel('Inference Time (ms)')
        ax5.set_title('Image Size vs Inference Time')
        ax5.grid(True, alpha=0.3)
        
        # 6. Image size vs detection count
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(size_results['sizes'], size_results['detection_counts'], 
                marker='D', linewidth=2, markersize=8, color='purple')
        ax6.set_xlabel('Image Size (pixels)')
        ax6.set_ylabel('Number of Detections')
        ax6.set_title('Image Size vs Detection Count')
        ax6.grid(True, alpha=0.3)
        
        # 7. FPS vs Image Size
        ax7 = plt.subplot(3, 3, 7)
        fps_values = [1000.0 / t for t in size_results['inference_times']]
        ax7.plot(size_results['sizes'], fps_values, 
                marker='o', linewidth=2, markersize=8, color='darkgreen')
        ax7.set_xlabel('Image Size (pixels)')
        ax7.set_ylabel('FPS')
        ax7.set_title('Image Size vs FPS\n(Frames Per Second)')
        ax7.grid(True, alpha=0.3)
        
        # 8. Class distribution
        class_counts, class_confidences = self.analyze_class_distribution(image)
        if class_counts:
            ax8 = plt.subplot(3, 3, 8)
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            bars = ax8.barh(classes, counts)
            ax8.set_xlabel('Number of Detections')
            ax8.set_title('Detected Object Classes')
            ax8.grid(True, alpha=0.3, axis='x')
            
            # Color bars by average confidence
            for i, (cls, bar) in enumerate(zip(classes, bars)):
                avg_conf = np.mean(class_confidences[cls])
                bar.set_color(plt.cm.RdYlGn(avg_conf))
        
        # 9. Detection result preview
        ax9 = plt.subplot(3, 3, 9)
        results = self.model(image, conf=0.25, verbose=False)
        annotated = results[0].plot()
        ax9.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        ax9.set_title(f'Detection Results\n(conf=0.25, {len(results[0].boxes) if results[0].boxes else 0} objects)')
        ax9.axis('off')
        
        plt.tight_layout()
        plt.savefig('yolo_performance_analysis.png', dpi=150, bbox_inches='tight')
        print("\nVisualization saved as: yolo_performance_analysis.png")
        plt.show()
    
    def print_performance_summary(self, image):
        """
        Print comprehensive performance summary
        
        Args:
            image: Test image
        """
        print("\n" + "="*60)
        print("YOLO PERFORMANCE ANALYSIS SUMMARY")
        print("="*60)
        
        # Speed benchmark
        speed_stats, _ = self.benchmark_inference_speed(image, num_runs=30)
        
        # Optimal parameters analysis
        print("\n--- Optimal Parameters Analysis ---")
        
        # Test different confidence thresholds
        print("\nRecommended Confidence Thresholds:")
        print("  0.25: Balanced (default) - good for general use")
        print("  0.50: Conservative - fewer false positives")
        print("  0.75: Strict - high confidence only")
        
        # Test different IoU thresholds
        print("\nRecommended IoU Thresholds (NMS):")
        print("  0.45: Standard - good balance")
        print("  0.50: Default - balanced suppression")
        print("  0.60: Loose - keeps more overlapping boxes")
        
        print("\n" + "="*60)


def main():
    """
    Main function for YOLO performance analysis
    """
    print("=== YOLO Performance Analysis ===")
    
    # Initialize analyzer
    analyzer = YOLOPerformanceAnalyzer()
    
    # Load test image
    test_image_path = 'yolo_test_scene.jpeg'
    if os.path.exists(test_image_path):
        test_image = cv2.imread(test_image_path)
        print(f"Loaded test image: {test_image_path}")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Creating sample test image...")
        # Create a blank image as fallback
        test_image = np.ones((640, 640, 3), dtype=np.uint8) * 200
        cv2.putText(test_image, "Add yolo_test_scene.jpeg for better analysis", 
                   (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    print(f"Image size: {test_image.shape[1]}x{test_image.shape[0]}")
    
    # Run comprehensive analysis
    try:
        # Visual analysis
        analyzer.visualize_performance_analysis(test_image)
        
        # Print summary
        analyzer.print_performance_summary(test_image)
        
        print("\n=== Analysis Complete ===")
        print("\nKey Takeaways:")
        print("1. Higher confidence threshold = fewer detections, higher precision")
        print("2. Lower IoU threshold = more aggressive NMS, fewer overlapping boxes")
        print("3. Larger image size = better accuracy but slower inference")
        print("4. YOLOv8n (nano) optimized for speed while maintaining good accuracy")
        print("5. Trade-off between speed and accuracy based on application needs")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
