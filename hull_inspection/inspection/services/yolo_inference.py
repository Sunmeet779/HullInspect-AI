"""
YOLO Inference Service for Hull Defect Detection

Two-Stage Detection Pipeline:
1. Stage 1: Detect boat/vessel using YOLO COCO pretrained model
2. Stage 2: Crop boat region and detect corrosion/defects using trained model

This ensures we only analyze actual boat surfaces, not background elements.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from django.conf import settings


class YOLOInferenceService:
    """Service for running two-stage YOLO inference on hull images."""
    
    # COCO class IDs for boats/vessels
    COCO_BOAT_CLASSES = {
        8: 'boat',
    }
    
    # Defect class mapping (from our trained model)
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
    
    # Colors for each class (BGR format) - High contrast colors
    CLASS_COLORS = {
        'boat': (255, 165, 0),                # Orange for boat bbox
        'corrosion_severe': (0, 0, 255),      # Red
        'corrosion_moderate': (0, 100, 255),  # Dark Orange
        'corrosion_mild': (0, 200, 200),      # Dark Yellow/Gold
        'paint_damage': (200, 0, 200),        # Dark Magenta
        'fouling': (0, 180, 0),               # Dark Green
        'crack': (255, 50, 50),               # Light Blue
        'dent': (200, 200, 0),                # Dark Cyan
        'surface_defect': (100, 100, 100),    # Dark Gray
    }
    
    # Severity mapping based on confidence
    SEVERITY_THRESHOLDS = {
        'critical': 0.85,
        'severe': 0.70,
        'moderate': 0.50,
        'mild': 0.0,
    }
    
    def __init__(self, model_path: Optional[str] = None, enable_boat_detection: bool = False):
        """
        Initialize the YOLO inference service.
        
        Args:
            model_path: Path to the trained YOLO model weights for defect detection.
                       If None, uses default from settings.
            enable_boat_detection: If True, first detect boat then analyze for defects.
                                  Default is False since dataset contains close-up hull images.
        """
        self.defect_model_path = model_path or self._find_best_model()
        self.defect_model = None
        self.boat_model = None  # COCO pretrained for boat detection
        self.enable_boat_detection = enable_boat_detection
        self.device = 'cpu'  # Use 'cuda' if GPU available
        # Backward compatibility
        self.model_path = self.defect_model_path
        self.model = None
    
    def _find_best_model(self) -> str:
        """Find the best trained model from multiple possible locations."""
        # Priority order for model search
        search_paths = [
            str(settings.YOLO_MODEL_PATH),  # ml_models/yolo/weights/best.pt
        ]
        
        # Also check training runs directory for latest trained model
        runs_path = settings.BASE_DIR / 'runs' / 'detect'
        if runs_path.exists():
            # Get most recent training run
            run_dirs = sorted(runs_path.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            for run_dir in run_dirs:
                best_pt = run_dir / 'weights' / 'best.pt'
                if best_pt.exists():
                    search_paths.insert(0, str(best_pt))  # Prioritize latest training
                    break
        
        # Return first existing model
        for path in search_paths:
            if os.path.exists(path):
                print(f"Found trained defect model: {path}")
                return path
        
        # Fallback to pretrained
        print("No trained defect model found")
        return None
        
    def load_models(self):
        """Load both YOLO models (boat detection and defect detection)."""
        try:
            from ultralytics import YOLO
            
            # Load COCO pretrained model for boat detection
            if self.enable_boat_detection:
                print("Loading COCO pretrained model for boat detection...")
                self.boat_model = YOLO('yolov8n.pt')
                self.boat_model.to(self.device)
                print("✅ Boat detection model loaded (COCO YOLOv8n)")
            
            # Load trained defect detection model
            if self.defect_model_path and os.path.exists(self.defect_model_path):
                print(f"Loading defect detection model: {self.defect_model_path}")
                self.defect_model = YOLO(self.defect_model_path)
                self.defect_model.to(self.device)
                self.model = self.defect_model  # Backward compatibility
                print("✅ Defect detection model loaded")
            else:
                print("⚠️ No trained defect model found, using COCO model only")
                self.defect_model = None
                
        except ImportError:
            raise ImportError(
                "ultralytics package not installed. "
                "Run: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO models: {str(e)}")
    
    def load_model(self):
        """Alias for load_models for backward compatibility."""
        self.load_models()
    
    def detect_boats(
        self, 
        image: np.ndarray,
        conf_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Stage 1: Detect boats in the image using COCO pretrained model.
        
        Args:
            image: Input image as numpy array
            conf_threshold: Confidence threshold for boat detection
            
        Returns:
            List of boat detections with bounding boxes
        """
        if self.boat_model is None:
            self.load_models()
        
        img_height, img_width = image.shape[:2]
        
        # Run COCO model inference
        results = self.boat_model(
            image,
            conf=conf_threshold,
            verbose=False
        )
        
        boats = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    # Check if it's a boat (class 8 in COCO)
                    if class_id == 8:  # boat class in COCO
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                        confidence = float(box.conf[0].cpu().numpy())
                        boats.append({
                            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                            'confidence': confidence,
                            'class_name': 'boat'
                        })
        
        # If no boat detected, treat entire image as the region of interest
        if not boats:
            print("⚠️ No boat detected - analyzing entire image")
            boats.append({
                'bbox': {'x1': 0, 'y1': 0, 'x2': img_width, 'y2': img_height},
                'confidence': 1.0,
                'class_name': 'full_image'
            })
        
        return boats
    
    def detect_defects_in_region(
        self,
        image: np.ndarray,
        region_bbox: Dict,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Dict]:
        """
        Stage 2: Detect defects within a specific region (boat bounding box).
        
        Args:
            image: Full image as numpy array
            region_bbox: Bounding box of the region to analyze
            conf_threshold: Confidence threshold for defect detection
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of defect detections
        """
        if self.defect_model is None:
            print("⚠️ No defect model loaded - skipping defect detection")
            return []
        
        # Crop the region
        x1, y1 = region_bbox['x1'], region_bbox['y1']
        x2, y2 = region_bbox['x2'], region_bbox['y2']
        
        # Add padding around the boat (10% on each side)
        img_h, img_w = image.shape[:2]
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        
        x1_padded = max(0, x1 - pad_x)
        y1_padded = max(0, y1 - pad_y)
        x2_padded = min(img_w, x2 + pad_x)
        y2_padded = min(img_h, y2 + pad_y)
        
        cropped_region = image[y1_padded:y2_padded, x1_padded:x2_padded]
        
        if cropped_region.size == 0:
            return []
        
        # Run defect detection on cropped region
        results = self.defect_model(
            cropped_region,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get coordinates in cropped image
                    cx1, cy1, cx2, cy2 = [float(v) for v in box.xyxy[0].cpu().numpy()]
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Convert back to original image coordinates
                    orig_x1 = int(cx1 + x1_padded)
                    orig_y1 = int(cy1 + y1_padded)
                    orig_x2 = int(cx2 + x1_padded)
                    orig_y2 = int(cy2 + y1_padded)
                    
                    # Calculate normalized coordinates (relative to full image)
                    x_center = float(((orig_x1 + orig_x2) / 2) / img_w)
                    y_center = float(((orig_y1 + orig_y2) / 2) / img_h)
                    width = float((orig_x2 - orig_x1) / img_w)
                    height = float((orig_y2 - orig_y1) / img_h)
                    
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
                            'x1': orig_x1,
                            'y1': orig_y1,
                            'x2': orig_x2,
                            'y2': orig_y2
                        },
                        'severity': str(severity),
                        'area_percent': float((width * height) * 100)
                    }
                    detections.append(detection)
        
        return detections
    
    def detect(
        self, 
        image_path: str,
        conf_threshold: float = 0.35,  # Higher threshold to reduce false positives
        iou_threshold: float = 0.45,
        save_annotated: bool = True,
        boat_conf_threshold: float = 0.3
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        Run two-stage detection on an image.
        
        Stage 1: Detect boat using COCO pretrained model
        Stage 2: Detect defects within boat region using trained model
        
        Args:
            image_path: Path to the input image
            conf_threshold: Confidence threshold for defect detections
            iou_threshold: IoU threshold for NMS
            save_annotated: Whether to save annotated image
            boat_conf_threshold: Confidence threshold for boat detection
            
        Returns:
            Tuple of (detections list, annotated image path)
        """
        if self.boat_model is None and self.defect_model is None:
            self.load_models()
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        img_height, img_width = image.shape[:2]
        all_detections = []
        boat_detections = []
        
        # Stage 1: Detect boats
        if self.enable_boat_detection and self.boat_model is not None:
            print("\n🚢 Stage 1: Detecting boats...")
            boat_detections = self.detect_boats(image, conf_threshold=boat_conf_threshold)
            print(f"   Found {len(boat_detections)} boat region(s)")
            
            for i, boat in enumerate(boat_detections):
                bbox = boat['bbox']
                print(f"   Boat {i+1}: {bbox['x1']},{bbox['y1']} -> {bbox['x2']},{bbox['y2']} "
                      f"(conf: {boat['confidence']:.2f})")
        else:
            # Analyze full image
            boat_detections = [{
                'bbox': {'x1': 0, 'y1': 0, 'x2': img_width, 'y2': img_height},
                'confidence': 1.0,
                'class_name': 'full_image'
            }]
        
        # Stage 2: Detect defects in each boat region
        print("\n🔍 Stage 2: Detecting corrosion/defects...")
        for boat in boat_detections:
            defects = self.detect_defects_in_region(
                image,
                boat['bbox'],
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            all_detections.extend(defects)
        
        print(f"   Found {len(all_detections)} defect(s)")
        
        # Create annotated image
        annotated_path = None
        if save_annotated:
            annotated_path = self._create_annotated_image(
                image, 
                all_detections,
                boat_detections if self.enable_boat_detection else [],
                image_path
            )
        
        return all_detections, annotated_path
    
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
        defect_detections: List[Dict],
        boat_detections: List[Dict],
        original_path: str
    ) -> str:
        """
        Create annotated image with boat and defect bounding boxes.
        
        Args:
            image: Original image as numpy array
            defect_detections: List of defect detection dictionaries
            boat_detections: List of boat detection dictionaries
            original_path: Path to original image
            
        Returns:
            Path to saved annotated image
        """
        annotated = image.copy()
        
        # Draw boat bounding boxes (Stage 1 results)
        for boat in boat_detections:
            if boat['class_name'] == 'full_image':
                continue  # Don't draw box around entire image
                
            bbox = boat['bbox']
            # Draw thick boat boundary
            cv2.rectangle(
                annotated,
                (bbox['x1'], bbox['y1']),
                (bbox['x2'], bbox['y2']),
                self.CLASS_COLORS.get('boat', (255, 165, 0)),
                3
            )
            
            # Label for boat
            label = f"BOAT DETECTED ({boat['confidence']:.0%})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (bbox['x1'], bbox['y1'] - text_height - 10),
                (bbox['x1'] + text_width + 10, bbox['y1']),
                self.CLASS_COLORS.get('boat', (255, 165, 0)),
                -1
            )
            
            cv2.putText(
                annotated,
                label,
                (bbox['x1'] + 5, bbox['y1'] - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        # Draw defect bounding boxes (Stage 2 results)
        for det in defect_detections:
            bbox = det['bbox_pixels']
            class_name = det['class_name']
            confidence = det['confidence']
            severity = det['severity']
            
            # Get color for this class
            color = self.CLASS_COLORS.get(class_name, (255, 255, 255))
            
            # Draw bounding box with thicker line
            cv2.rectangle(
                annotated,
                (bbox['x1'], bbox['y1']),
                (bbox['x2'], bbox['y2']),
                color,
                3  # Thicker line
            )
            
            # Create label with short class name
            short_name = class_name.replace('corrosion_', '').replace('_', ' ').upper()
            label = f"{short_name}: {confidence:.0%}"
            
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw dark background for better visibility
            cv2.rectangle(
                annotated,
                (bbox['x1'], bbox['y1'] - text_height - 12),
                (bbox['x1'] + text_width + 10, bbox['y1']),
                (0, 0, 0),  # Black background
                -1
            )
            
            # Draw colored border on label background
            cv2.rectangle(
                annotated,
                (bbox['x1'], bbox['y1'] - text_height - 12),
                (bbox['x1'] + text_width + 10, bbox['y1']),
                color,
                2
            )
            
            # Draw label text in white (high contrast on black bg)
            cv2.putText(
                annotated,
                label,
                (bbox['x1'] + 5, bbox['y1'] - 5),
                font,
                font_scale,
                (255, 255, 255),  # White text
                thickness
            )
            
            # Draw severity indicator with black outline for visibility
            severity_text = f"[{severity.upper()}]"
            severity_color = {
                'critical': (0, 0, 255),      # Red
                'severe': (0, 100, 255),      # Orange
                'moderate': (0, 180, 255),    # Light Orange
                'mild': (0, 200, 0),          # Green
            }.get(severity, (255, 255, 255))
            
            # Black outline for severity text
            cv2.putText(
                annotated,
                severity_text,
                (bbox['x1'], bbox['y2'] + 18),
                font,
                font_scale * 0.8,
                (0, 0, 0),  # Black outline
                thickness + 2
            )
            # Colored severity text
            cv2.putText(
                annotated,
                severity_text,
                (bbox['x1'], bbox['y2'] + 18),
                font,
                font_scale * 0.8,
                severity_color,
                thickness
            )
        
        # Add summary text at top of image with high visibility
        num_boats = len([b for b in boat_detections if b['class_name'] != 'full_image'])
        if num_boats > 0:
            summary_text = f"Boats: {num_boats} | Defects: {len(defect_detections)}"
        else:
            summary_text = f"Hull Inspection | Defects Found: {len(defect_detections)}"
        
        # Draw black background for summary
        (text_w, text_h), _ = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(annotated, (5, 5), (text_w + 20, text_h + 20), (0, 0, 0), -1)
        cv2.rectangle(annotated, (5, 5), (text_w + 20, text_h + 20), (0, 255, 0), 2)
        
        cv2.putText(
            annotated,
            summary_text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        # Save annotated image
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        output_dir = os.path.join(settings.MEDIA_ROOT, 'uploads', 'annotated')
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
