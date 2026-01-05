"""
YOLO Inference Service for Hull Defect Detection

This service handles:
- Loading the trained YOLO model
- Running inference on uploaded images
- Processing and formatting detection results
- Creating annotated images with bounding boxes
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from django.conf import settings


class YOLOInferenceService:
    """Service for running YOLO inference on hull images."""
    
    # Defect class mapping
    CLASS_NAMES = {
        0: 'corrosion_severe',
        1: 'corrosion_moderate',
        2: 'corrosion_mild',
        3: 'paint_damage',
        4: 'fouling',
        5: 'crack',
        6: 'dent',
        7: 'surface_defect',
    }
    
    # Colors for each class (BGR format)
    CLASS_COLORS = {
        'corrosion_severe': (0, 0, 255),      # Red
        'corrosion_moderate': (0, 128, 255),  # Orange
        'corrosion_mild': (0, 255, 255),      # Yellow
        'paint_damage': (255, 0, 255),        # Magenta
        'fouling': (0, 255, 0),               # Green
        'crack': (255, 0, 0),                 # Blue
        'dent': (255, 255, 0),                # Cyan
        'surface_defect': (128, 128, 128),    # Gray
    }
    
    # Severity mapping based on confidence
    SEVERITY_THRESHOLDS = {
        'critical': 0.85,
        'severe': 0.70,
        'moderate': 0.50,
        'mild': 0.0,
    }
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the YOLO inference service.
        
        Args:
            model_path: Path to the trained YOLO model weights.
                       If None, uses default from settings.
        """
        self.model_path = model_path or str(settings.YOLO_MODEL_PATH)
        self.model = None
        self.device = 'cpu'  # Use 'cuda' if GPU available
        
    def load_model(self):
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO
            
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
            else:
                # Use pretrained model if custom weights not available
                print(f"Custom model not found at {self.model_path}")
                print("Using pretrained YOLOv8 model...")
                self.model = YOLO('yolov8n.pt')
                
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}")
            
        except ImportError:
            raise ImportError(
                "ultralytics package not installed. "
                "Run: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")
    
    def detect(
        self, 
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        save_annotated: bool = True
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        Run detection on an image.
        
        Args:
            image_path: Path to the input image
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            save_annotated: Whether to save annotated image
            
        Returns:
            Tuple of (detections list, annotated image path)
        """
        if self.model is None:
            self.load_model()
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        img_height, img_width = image.shape[:2]
        
        # Run inference
        results = self.model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # Process results
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get coordinates and convert to Python native types
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().numpy()]
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Convert to normalized center format (ensure Python float)
                    x_center = float(((x1 + x2) / 2) / img_width)
                    y_center = float(((y1 + y2) / 2) / img_height)
                    width = float((x2 - x1) / img_width)
                    height = float((y2 - y1) / img_height)
                    
                    # Get class name
                    class_name = self.CLASS_NAMES.get(
                        class_id, 
                        result.names.get(class_id, f'class_{class_id}')
                    )
                    
                    # Determine severity
                    severity = self._get_severity(confidence, class_name)
                    
                    detection = {
                        'class_id': int(class_id),
                        'class_name': str(class_name),
                        'confidence': float(confidence),
                        'x_center': float(x_center),
                        'y_center': float(y_center),
                        'width': float(width),
                        'height': float(height),
                        'bbox_pixels': {
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2)
                        },
                        'severity': str(severity),
                        'area_percent': float((width * height) * 100)
                    }
                    detections.append(detection)
        
        # Create annotated image
        annotated_path = None
        if save_annotated and detections:
            annotated_path = self._create_annotated_image(
                image, 
                detections, 
                image_path
            )
        
        return detections, annotated_path
    
    def _get_severity(self, confidence: float, class_name: str) -> str:
        """Determine severity based on confidence and class."""
        # Severe corrosion is always critical/severe
        if 'severe' in class_name.lower():
            return 'critical' if confidence > 0.7 else 'severe'
        
        # Cracks are high priority
        if 'crack' in class_name.lower():
            return 'severe' if confidence > 0.6 else 'moderate'
        
        # Standard severity mapping
        for severity, threshold in self.SEVERITY_THRESHOLDS.items():
            if confidence >= threshold:
                return severity
        
        return 'mild'
    
    def _create_annotated_image(
        self, 
        image: np.ndarray, 
        detections: List[Dict],
        original_path: str
    ) -> str:
        """
        Create annotated image with bounding boxes and labels.
        
        Args:
            image: Original image as numpy array
            detections: List of detection dictionaries
            original_path: Path to original image
            
        Returns:
            Path to saved annotated image
        """
        annotated = image.copy()
        
        for det in detections:
            bbox = det['bbox_pixels']
            class_name = det['class_name']
            confidence = det['confidence']
            severity = det['severity']
            
            # Get color for this class
            color = self.CLASS_COLORS.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(
                annotated,
                (bbox['x1'], bbox['y1']),
                (bbox['x2'], bbox['y2']),
                color,
                2
            )
            
            # Create label
            label = f"{class_name}: {confidence:.2f}"
            severity_label = f"[{severity.upper()}]"
            
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (bbox['x1'], bbox['y1'] - text_height - 10),
                (bbox['x1'] + text_width + 10, bbox['y1']),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (bbox['x1'] + 5, bbox['y1'] - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
            
            # Draw severity indicator
            severity_color = {
                'critical': (0, 0, 255),
                'severe': (0, 128, 255),
                'moderate': (0, 255, 255),
                'mild': (0, 255, 0),
            }.get(severity, (255, 255, 255))
            
            cv2.putText(
                annotated,
                severity_label,
                (bbox['x1'], bbox['y2'] + 15),
                font,
                font_scale * 0.8,
                severity_color,
                thickness
            )
        
        # Save annotated image
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        output_dir = os.path.join(settings.MEDIA_ROOT, 'annotated')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
        cv2.imwrite(output_path, annotated)
        
        return output_path
    
    def get_detection_summary(self, detections: List[Dict]) -> Dict:
        """
        Generate a summary of detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Summary dictionary with counts and statistics
        """
        if not detections:
            return {
                'total_defects': 0,
                'by_class': {},
                'by_severity': {},
                'average_confidence': 0,
                'total_affected_area': 0
            }
        
        summary = {
            'total_defects': len(detections),
            'by_class': {},
            'by_severity': {},
            'average_confidence': 0,
            'total_affected_area': 0
        }
        
        total_confidence = 0
        total_area = 0
        
        for det in detections:
            class_name = det['class_name']
            severity = det['severity']
            
            # Count by class
            if class_name not in summary['by_class']:
                summary['by_class'][class_name] = 0
            summary['by_class'][class_name] += 1
            
            # Count by severity
            if severity not in summary['by_severity']:
                summary['by_severity'][severity] = 0
            summary['by_severity'][severity] += 1
            
            # Sum confidence and area
            total_confidence += det['confidence']
            total_area += det.get('area_percent', 0)
        
        summary['average_confidence'] = total_confidence / len(detections)
        summary['total_affected_area'] = total_area
        
        return summary


# For testing
if __name__ == '__main__':
    service = YOLOInferenceService()
    
    # Test with a sample image
    test_image = '/path/to/test/image.jpg'
    if os.path.exists(test_image):
        detections, annotated_path = service.detect(test_image)
        print(f"Found {len(detections)} defects")
        print(f"Annotated image saved to: {annotated_path}")
