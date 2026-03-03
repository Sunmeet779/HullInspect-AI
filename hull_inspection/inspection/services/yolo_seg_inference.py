"""
YOLO11-Seg Inference Service for Hull Corrosion Segmentation

Instance Segmentation Pipeline:
- Detects corrosion regions with pixel-level masks
- Calculates precise corrosion area (not bounding box area)
- Assigns severity based on actual corroded area percentage
- Generates annotated images with colored mask overlays

This replaces the previous bounding-box detection approach.
Corrosion has irregular shapes that bounding boxes cannot capture.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from django.conf import settings


class YOLOSegInferenceService:
    """Service for running YOLO11-Seg inference on hull images."""
    
    # Single class — the MaVeCoDD dataset labels corrosion only
    CLASS_NAMES = {
        0: 'corrosion',
    }
    
    # Severity classification based on actual corroded area percentage
    SEVERITY_BY_AREA = {
        'critical': 5.0,    # > 5% of image area
        'severe': 2.0,      # > 2%
        'moderate': 0.5,    # > 0.5%
        'mild': 0.0,        # anything smaller
    }
    
    # Overlay colors (BGR) for visualization
    MASK_COLORS = {
        'critical': (0, 0, 220),      # Red
        'severe': (0, 80, 255),       # Orange
        'moderate': (0, 200, 255),    # Yellow
        'mild': (0, 220, 0),          # Green
    }
    
    MASK_ALPHA = 0.45  # Transparency for mask overlay
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.70,
    ):
        """
        Initialize the segmentation inference service.
        
        Args:
            model_path: Path to trained YOLO11-seg model weights (.pt).
                       If None, searches for the best available model.
            conf_threshold: Default confidence threshold
            iou_threshold: Default IoU threshold for NMS
        """
        self.model_path = model_path or self._find_best_model()
        self.model = None
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = self._detect_device()
        
    def _detect_device(self) -> str:
        """Detect the best available device."""
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            elif torch.cuda.is_available():
                return 'cuda'
        except ImportError:
            pass
        return 'cpu'
    
    def _find_best_model(self) -> Optional[str]:
        """Find the best trained segmentation model."""
        search_paths = []
        
        # Check ml_models/yolo/weights/ first
        primary = settings.BASE_DIR / 'ml_models' / 'yolo' / 'weights' / 'best_seg.pt'
        search_paths.append(str(primary))
        
        # Check segmentation training runs
        runs_path = settings.BASE_DIR / 'runs' / 'segment'
        if runs_path.exists():
            run_dirs = sorted(runs_path.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            for run_dir in run_dirs:
                best_pt = run_dir / 'weights' / 'best.pt'
                if best_pt.exists():
                    search_paths.insert(0, str(best_pt))
                    break
        
        # Also check old detection runs as fallback
        det_runs = settings.BASE_DIR / 'runs' / 'detect'
        if det_runs.exists():
            run_dirs = sorted(det_runs.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            for run_dir in run_dirs:
                best_pt = run_dir / 'weights' / 'best.pt'
                if best_pt.exists():
                    search_paths.append(str(best_pt))
                    break
        
        # Also check the old settings path
        search_paths.append(str(settings.YOLO_MODEL_PATH))
        
        for path in search_paths:
            if os.path.exists(path):
                print(f"Found model: {path}")
                return path
        
        print("⚠️ No trained model found")
        return None
        
    def load_model(self):
        """Load the YOLO segmentation model."""
        try:
            from ultralytics import YOLO
            
            if self.model_path and os.path.exists(self.model_path):
                print(f"Loading segmentation model: {self.model_path}")
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
                print(f"✅ Model loaded on {self.device}")
            else:
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}. "
                    "Train a model first with train_seg.py"
                )
                
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics>=8.1.0")
    
    def detect(
        self,
        image_path: str,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        save_annotated: bool = True,
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        Run segmentation inference on an image.
        
        Args:
            image_path: Path to the input image
            conf_threshold: Confidence threshold (overrides default)
            iou_threshold: IoU threshold (overrides default)
            save_annotated: Whether to save annotated image
            
        Returns:
            Tuple of (detections list, annotated image path or None)
        """
        if self.model is None:
            self.load_model()
        
        conf = conf_threshold or self.conf_threshold
        iou = iou_threshold or self.iou_threshold
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        img_h, img_w = image.shape[:2]
        total_pixels = img_h * img_w
        
        print(f"\n🔍 Running segmentation inference...")
        print(f"   Image: {os.path.basename(image_path)} ({img_w}x{img_h})")
        print(f"   Confidence threshold: {conf}")
        
        # Run inference
        results = self.model(
            image,
            conf=conf,
            iou=iou,
            verbose=False,
            retina_masks=True,  # High-res masks matching original image size
        )
        
        detections = []
        all_masks = []
        
        for result in results:
            boxes = result.boxes
            masks = result.masks
            
            if boxes is None or masks is None:
                continue
            
            for i in range(len(boxes)):
                # Bounding box
                x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].cpu().numpy()]
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.CLASS_NAMES.get(class_id, f'class_{class_id}')
                
                # Segmentation mask
                mask = masks.data[i].cpu().numpy()  # Binary mask (H, W)
                
                # Resize mask to original image size if needed
                if mask.shape != (img_h, img_w):
                    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                
                mask_binary = (mask > 0.5).astype(np.uint8)
                
                # Calculate precise area from mask (not bounding box)
                mask_pixels = np.sum(mask_binary)
                area_percent = float(mask_pixels / total_pixels * 100)
                
                # Determine severity from actual corroded area
                severity = self._get_severity_from_area(area_percent)
                
                # Normalized bbox
                x_center = float(((x1 + x2) / 2) / img_w)
                y_center = float(((y1 + y2) / 2) / img_h)
                width = float((x2 - x1) / img_w)
                height = float((y2 - y1) / img_h)
                
                # Get polygon contour from mask (for storage/annotation)
                contours, _ = cv2.findContours(
                    mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Convert contours to normalized polygon points
                polygon_points = []
                for contour in contours:
                    if cv2.contourArea(contour) < 50:
                        continue
                    approx = cv2.approxPolyDP(contour, 0.005 * cv2.arcLength(contour, True), True)
                    pts = []
                    for pt in approx:
                        pts.append([float(pt[0][0]) / img_w, float(pt[0][1]) / img_h])
                    if len(pts) >= 3:
                        polygon_points.append(pts)
                
                detection = {
                    'class_id': int(class_id),
                    'class_name': str(class_name),
                    'confidence': float(confidence),
                    'x_center': float(x_center),
                    'y_center': float(y_center),
                    'width': float(width),
                    'height': float(height),
                    'bbox_pixels': {
                        'x1': int(x1), 'y1': int(y1),
                        'x2': int(x2), 'y2': int(y2),
                    },
                    'severity': str(severity),
                    'area_percent': float(area_percent),
                    'mask_pixels': int(mask_pixels),
                    'polygon_points': polygon_points,  # For JSON storage
                }
                detections.append(detection)
                all_masks.append((mask_binary, severity))
        
        # Sort by area (largest first) — keep masks in sync with detections
        paired = sorted(
            zip(detections, all_masks),
            key=lambda p: p[0]['area_percent'],
            reverse=True,
        )
        if paired:
            detections, all_masks = list(zip(*paired))
            detections = list(detections)
            all_masks = list(all_masks)

        # Filter out noise: drop regions with fewer than 100 mask pixels
        # or less than 0.05% area (likely false positives on small images)
        MIN_MASK_PX = 100
        MIN_AREA_PCT = 0.05
        MAX_DETECTIONS = 12       # cap detections for readability
        filtered_d, filtered_m = [], []
        for det, msk in zip(detections, all_masks):
            if det['mask_pixels'] >= MIN_MASK_PX and det['area_percent'] >= MIN_AREA_PCT:
                filtered_d.append(det)
                filtered_m.append(msk)
        detections, all_masks = filtered_d[:MAX_DETECTIONS], filtered_m[:MAX_DETECTIONS]
        if len(filtered_d) > MAX_DETECTIONS:
            print(f"   ⚠ Capped detections from {len(filtered_d)} to {MAX_DETECTIONS}")

        # Calculate total corrosion area using UNION of all masks
        # (naive sum double-counts overlapping pixels)
        if all_masks:
            union_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for mask_binary, _ in all_masks:
                union_mask = np.bitwise_or(union_mask, mask_binary)
            union_pixels = int(np.sum(union_mask))
            total_corrosion_area = float(union_pixels / total_pixels * 100)
        else:
            union_pixels = 0
            total_corrosion_area = 0.0
        
        # Store union area for use by summary/annotation
        self._last_union_area = total_corrosion_area
        
        print(f"   Found {len(detections)} corrosion region(s)")
        print(f"   Total corroded area: {total_corrosion_area:.2f}% of image (union of all masks)")
        
        # Create annotated image
        annotated_path = None
        if save_annotated and detections:
            annotated_path = self._create_annotated_image(
                image, detections, all_masks, image_path
            )
        
        return detections, annotated_path
    
    def _get_severity_from_area(self, area_percent: float) -> str:
        """Determine severity from actual mask area percentage."""
        for severity, threshold in self.SEVERITY_BY_AREA.items():
            if area_percent >= threshold:
                return severity
        return 'mild'
    
    # Distinct color palette (BGR) — each detection gets its own color
    DETECTION_COLORS = [
        (0, 0, 230),      # Red
        (230, 160, 0),     # Blue
        (0, 200, 0),       # Green
        (0, 180, 255),     # Orange
        (200, 0, 200),     # Magenta
        (255, 255, 0),     # Cyan
        (80, 127, 255),    # Coral
        (0, 255, 255),     # Yellow
        (255, 0, 128),     # Purple
        (128, 255, 0),     # Spring green
    ]

    # Minimum annotation canvas size — tiny uploads get upscaled so labels are readable
    MIN_ANNOTATE_DIM = 800

    def _create_annotated_image(
        self,
        image: np.ndarray,
        detections: List[Dict],
        masks_with_severity: List[Tuple[np.ndarray, str]],
        original_path: str,
    ) -> str:
        """
        Create annotated image with coloured segmentation overlays.

        Handles images of any size:
        - Tiny images are upscaled to at least MIN_ANNOTATE_DIM before
          drawing so labels remain readable.
        - All font sizes, line widths, and padding scale with image size.
        - Masks are blended individually (largest first, smallest last)
          so smaller regions always show on top.
        """
        img_h, img_w = image.shape[:2]
        n = len(detections)

        # ── 0.  Upscale tiny images so annotations are legible ─────────
        scale = 1.0
        short_side = min(img_h, img_w)
        if short_side < self.MIN_ANNOTATE_DIM:
            scale = self.MIN_ANNOTATE_DIM / short_side
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            # Upscale every mask to match
            new_masks = []
            for mask_bin, sev in masks_with_severity:
                m = cv2.resize(mask_bin, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                new_masks.append((m, sev))
            masks_with_severity = new_masks

            # Adjust bbox pixel coords
            for det in detections:
                bb = det['bbox_pixels']
                bb['x1'] = int(bb['x1'] * scale)
                bb['y1'] = int(bb['y1'] * scale)
                bb['x2'] = int(bb['x2'] * scale)
                bb['y2'] = int(bb['y2'] * scale)

            img_h, img_w = new_h, new_w

        annotated = image.copy()

        # ── Adaptive sizes based on image dimensions ───────────────────
        ref = max(img_h, img_w)
        contour_thick = max(1, int(ref / 500))
        bbox_thick    = max(1, int(ref / 700))
        fs            = max(0.30, ref / 2400)        # font scale
        font_thick    = max(1, int(ref / 1200))
        pad           = max(2, int(ref / 400))
        font          = cv2.FONT_HERSHEY_SIMPLEX

        # ── 1.  Per-region mask overlay (largest first) ────────────────
        draw_order = sorted(range(n), key=lambda i: detections[i]['area_percent'], reverse=True)

        for i in draw_order:
            mask_bin = masks_with_severity[i][0]
            color = self.DETECTION_COLORS[i % len(self.DETECTION_COLORS)]
            colour_layer = np.full_like(annotated, color, dtype=np.uint8)
            mask_bool = mask_bin > 0
            annotated[mask_bool] = cv2.addWeighted(
                colour_layer, 0.35, annotated, 0.65, 0
            )[mask_bool]

        # ── 2.  Contour outlines (smallest on top) ────────────────────
        for i in reversed(draw_order):
            mask_bin = masks_with_severity[i][0]
            color = self.DETECTION_COLORS[i % len(self.DETECTION_COLORS)]
            contours, _ = cv2.findContours(
                mask_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(annotated, contours, -1, color, contour_thick)
            cv2.drawContours(annotated, contours, -1, (255, 255, 255), max(1, contour_thick // 2))

        # ── 3.  Summary bar (draw early so labels avoid it) ────────────
        total_area = getattr(self, '_last_union_area',
                             min(sum(d['area_percent'] for d in detections), 100.0))
        overall_sev = self._get_overall_severity(detections)
        summary = (f"Corrosion: {n} regions | "
                   f"Total area: {total_area:.1f}% | {overall_sev.upper()}")

        sum_fs = max(0.35, ref / 2000)
        (sw, sh), _ = cv2.getTextSize(summary, font, sum_fs, 2)
        bar_h = sh + int(pad * 3)
        bar_rect = (0, 0, sw + 2 * pad + 10, bar_h)
        cv2.rectangle(annotated, (bar_rect[0], bar_rect[1]),
                      (bar_rect[2], bar_rect[3]), (0, 0, 0), -1)
        cv2.putText(annotated, summary, (pad + 4, sh + pad),
                    font, sum_fs, (0, 255, 0), max(1, font_thick), cv2.LINE_AA)

        # ── 4.  Compact labels at each bbox ───────────────────────────
        # Seed with bar rect so labels never hide behind the summary bar
        used_rects: List[Tuple[int, int, int, int]] = [bar_rect]

        for i in range(n):
            det = detections[i]
            color = self.DETECTION_COLORS[i % len(self.DETECTION_COLORS)]
            bb = det['bbox_pixels']
            conf = det['confidence']
            area = det['area_percent']
            sev  = det['severity'][0].upper()

            # Thin bounding box
            cv2.rectangle(annotated, (bb['x1'], bb['y1']),
                          (bb['x2'], bb['y2']), color, bbox_thick)

            label = f"#{i+1} {conf:.0%} {area:.1f}% [{sev}]"
            (tw, tht), _ = cv2.getTextSize(label, font, fs, font_thick)

            lx = max(bb['x1'], 2)
            ly = bb['y1'] - pad
            if ly - tht - pad < 0:
                ly = bb['y1'] + tht + pad + 4

            # Nudge down if overlapping a previous label
            rect = (lx, ly - tht - pad, lx + tw + 2 * pad, ly + pad)
            for _ in range(n):
                if not any(
                    not (rect[2] < pr[0] or rect[0] > pr[2] or rect[3] < pr[1] or rect[1] > pr[3])
                    for pr in used_rects
                ):
                    break
                ly += tht + pad + 2
                if ly + pad > img_h:
                    ly = tht + pad + 4
                    lx = min(lx + tw + pad, img_w - tw - pad)
                rect = (lx, ly - tht - pad, lx + tw + 2 * pad, ly + pad)

            used_rects.append(rect)

            cv2.rectangle(annotated, (lx - 1, ly - tht - pad),
                          (lx + tw + 2 * pad, ly + pad), (0, 0, 0), -1)
            cv2.rectangle(annotated, (lx - 1, ly - tht - pad),
                          (lx + tw + 2 * pad, ly + pad), color, max(1, bbox_thick))
            cv2.putText(annotated, label, (lx + pad, ly),
                        font, fs, (255, 255, 255), font_thick, cv2.LINE_AA)

        # ── 5.  Compact legend (bottom-left) ──────────────────────────
        leg_fs = max(0.28, ref / 3200)
        (_, leg_h), _ = cv2.getTextSize("#", font, leg_fs, 1)
        leg_step = leg_h + pad + 2
        leg_y = img_h - pad - n * leg_step
        if leg_y < bar_h + 10:
            leg_y = bar_h + 10

        for i in range(n):
            color = self.DETECTION_COLORS[i % len(self.DETECTION_COLORS)]
            sev = detections[i]['severity']
            area = detections[i]['area_percent']
            txt = f"#{i+1} {sev} {area:.1f}%"
            sq = max(6, int(ref / 200))
            cv2.rectangle(annotated, (pad, leg_y), (pad + sq, leg_y + sq), color, -1)
            cv2.putText(annotated, txt, (pad + sq + 4, leg_y + sq - 1),
                        font, leg_fs, (255, 255, 255), 1, cv2.LINE_AA)
            leg_y += leg_step

        # ── 6.  Save ──────────────────────────────────────────────────
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        output_dir = os.path.join(settings.MEDIA_ROOT, 'uploads', 'annotated')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}_seg_annotated.jpg")
        cv2.imwrite(output_path, annotated)

        print(f"   Annotated image saved: {output_path}")
        return output_path
    
    def _get_overall_severity(self, detections: List[Dict]) -> str:
        """Get overall severity from all detections (uses union area)."""
        if not detections:
            return 'good'
        
        # Use union area if available, otherwise cap at 100%
        total_area = getattr(self, '_last_union_area', min(sum(d['area_percent'] for d in detections), 100.0))
        
        # Overall severity based on total corroded area
        if total_area > 15.0:
            return 'critical'
        elif total_area > 8.0:
            return 'severe'
        elif total_area > 3.0:
            return 'moderate'
        elif total_area > 0.5:
            return 'mild'
        return 'good'
    
    def get_detection_summary(self, detections: List[Dict]) -> Dict:
        """
        Generate a summary of detections with area-based analysis.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Summary dictionary
        """
        if not detections:
            return {
                'total_regions': 0,
                'total_area_percent': 0.0,
                'by_severity': {},
                'average_confidence': 0.0,
                'overall_severity': 'good',
                'largest_region_area': 0.0,
            }
        
        by_severity = {}
        for det in detections:
            sev = det['severity']
            if sev not in by_severity:
                by_severity[sev] = {'count': 0, 'sum_area': 0.0}
            by_severity[sev]['count'] += 1
            by_severity[sev]['sum_area'] += det['area_percent']
        
        # Use union area if available, otherwise cap at 100%
        total_area = getattr(self, '_last_union_area', min(sum(d['area_percent'] for d in detections), 100.0))
        avg_conf = sum(d['confidence'] for d in detections) / len(detections)
        
        return {
            'total_regions': len(detections),
            'total_area_percent': round(total_area, 2),
            'by_severity': by_severity,
            'average_confidence': round(avg_conf, 4),
            'overall_severity': self._get_overall_severity(detections),
            'largest_region_area': round(max(d['area_percent'] for d in detections), 2),
        }


# Backward-compatible alias
YOLOInferenceService = YOLOSegInferenceService


# For testing
if __name__ == '__main__':
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python yolo_seg_inference.py <image_path>")
        sys.exit(1)
    
    # Quick test outside Django
    class MockSettings:
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        MEDIA_ROOT = BASE_DIR / 'media'
        YOLO_MODEL_PATH = BASE_DIR / 'ml_models' / 'yolo' / 'weights' / 'best.pt'
    
    settings = MockSettings()
    
    service = YOLOSegInferenceService()
    detections, annotated_path = service.detect(sys.argv[1])
    
    summary = service.get_detection_summary(detections)
    print(f"\nSummary: {json.dumps(summary, indent=2)}")
