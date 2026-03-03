"""
Dataset Preparation Script for YOLO Instance Segmentation Training

Converts the MaVeCoDD dataset (Marine Vessel Hull Corrosion in Dry-Dock)
from colored mask labels to YOLO polygon segmentation format.

KEY DIFFERENCE from prepare_dataset.py (bounding box version):
- Instead of converting masks → bounding boxes, we convert masks → polygons
- This preserves the irregular shape of corrosion regions
- Uses a single "corrosion" class (what the dataset actually labels)
- Produces YOLO-seg format: class_id x1 y1 x2 y2 ... xN yN (normalized)

Dataset Structure Expected:
    ry392rp8cj-1/
    ├── HiRes/
    │   ├── labeled/  (hull001_label.jpg, hull002_label.jpg, ...)
    │   └── raw/      (hull001.jpg, hull002.jpg, ...)
    └── LoRes/
        ├── labeled/
        └── raw/

Output Structure:
    datasets/
    └── hull_defects_seg/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── labels/
        │   ├── train/
        │   └── val/
        └── data.yaml
"""
import os
import cv2
import numpy as np
import shutil
import yaml
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse


class SegDatasetPreparer:
    """
    Prepares hull defect dataset for YOLO segmentation training.
    Converts colored mask images to YOLO polygon format.
    """
    
    # Single class - the dataset only truly labels corrosion
    CLASSES = {
        0: 'corrosion',
    }
    
    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        min_contour_area: int = 150,
        epsilon_factor: float = 0.005,
        min_polygon_points: int = 4,
    ):
        """
        Initialize the segmentation dataset preparer.
        
        Args:
            source_dir: Path to the MaVeCoDD dataset root (ry392rp8cj-1 folder)
            output_dir: Path to output YOLO-seg formatted dataset
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            min_contour_area: Minimum contour area in pixels to keep
            epsilon_factor: Polygon approximation tolerance (fraction of perimeter)
            min_polygon_points: Minimum number of polygon points to keep
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.min_contour_area = min_contour_area
        self.epsilon_factor = epsilon_factor
        self.min_polygon_points = min_polygon_points
        
        self.setup_directories()
        
    def setup_directories(self):
        """Create the YOLO dataset directory structure."""
        for split in ['train', 'val']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    def get_image_pairs(self) -> List[Tuple[Path, Path]]:
        """Get pairs of (raw_image, labeled_image) from the dataset."""
        pairs = []
        
        for resolution in ['HiRes', 'LoRes']:
            raw_dir = self.source_dir / resolution / 'raw'
            labeled_dir = self.source_dir / resolution / 'labeled'
            
            if not raw_dir.exists():
                print(f"  Skipping {resolution}: raw dir not found")
                continue
            
            for raw_path in sorted(raw_dir.glob('*.jpg')):
                base_name = raw_path.stem
                labeled_path = labeled_dir / f"{base_name}_label.jpg"
                
                if labeled_path.exists():
                    pairs.append((raw_path, labeled_path))
                else:
                    print(f"  Warning: No label found for {raw_path.name}")
        
        return pairs
    
    def mask_to_polygons(
        self,
        label_image: np.ndarray,
    ) -> List[List[float]]:
        """
        Convert a labeled mask image to normalized polygon coordinates.
        
        The MaVeCoDD dataset uses colored regions on a dark/black background
        to indicate corrosion. We extract these regions as polygons.
        
        Args:
            label_image: BGR label image
            
        Returns:
            List of polygons, each polygon is [x1, y1, x2, y2, ..., xN, yN]
            with coordinates normalized to [0, 1]
        """
        img_h, img_w = label_image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY)
        
        # Also check for colored regions (non-gray pixels)
        hsv = cv2.cvtColor(label_image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        # Binary mask: regions that are either bright or have saturation (color)
        mask = np.logical_or(gray > 20, saturation > 30).astype(np.uint8) * 255
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_contour_area:
                continue
            
            # Approximate polygon to reduce point count while preserving shape
            perimeter = cv2.arcLength(contour, True)
            epsilon = self.epsilon_factor * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Ensure minimum number of points
            if len(approx) < self.min_polygon_points:
                continue
            
            # Convert to normalized coordinates
            polygon = []
            for point in approx:
                x = float(point[0][0]) / img_w
                y = float(point[0][1]) / img_h
                # Clamp to [0, 1]
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                polygon.extend([x, y])
            
            polygons.append(polygon)
        
        return polygons
    
    def save_yolo_seg_annotation(
        self,
        polygons: List[List[float]],
        output_path: Path,
        class_id: int = 0,
    ):
        """
        Save polygons in YOLO segmentation format.
        
        YOLO-seg format: class_id x1 y1 x2 y2 ... xN yN
        All coordinates are normalized (0-1).
        
        Args:
            polygons: List of polygons
            output_path: Path to save the annotation file
            class_id: Class ID (0 for corrosion)
        """
        with open(output_path, 'w') as f:
            for polygon in polygons:
                coords = ' '.join(f'{c:.6f}' for c in polygon)
                f.write(f"{class_id} {coords}\n")
    
    def prepare_dataset(self, seed: int = 42):
        """
        Main method to prepare the complete segmentation dataset.
        
        Args:
            seed: Random seed for reproducible splits
        """
        print("=" * 70)
        print("Hull Defect SEGMENTATION Dataset Preparation")
        print("=" * 70)
        print(f"Source: {self.source_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Min contour area: {self.min_contour_area} px")
        print(f"Polygon epsilon: {self.epsilon_factor}")
        print()
        
        # Get all image pairs
        pairs = self.get_image_pairs()
        print(f"\nFound {len(pairs)} image pairs")
        
        if len(pairs) == 0:
            print("No image pairs found. Check your dataset path.")
            return
        
        # Shuffle and split
        random.seed(seed)
        random.shuffle(pairs)
        
        n_total = len(pairs)
        n_train = int(n_total * self.train_ratio)
        
        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train:]
        
        print(f"Split: Train={len(train_pairs)}, Val={len(val_pairs)}")
        
        # Process each split
        stats = {}
        
        for split, split_pairs in [('train', train_pairs), ('val', val_pairs)]:
            print(f"\nProcessing {split} split...")
            
            total_polygons = 0
            total_points = 0
            images_with_annotations = 0
            
            for raw_path, label_path in split_pairs:
                # Read label image
                label_img = cv2.imread(str(label_path))
                if label_img is None:
                    print(f"  Error reading label: {label_path}")
                    continue
                
                # Convert mask to polygons
                polygons = self.mask_to_polygons(label_img)
                
                if not polygons:
                    print(f"  Warning: No polygons found in {label_path.name}")
                    # Still include the image with empty label file for background
                
                # Copy raw image
                dest_img = self.output_dir / 'images' / split / raw_path.name
                shutil.copy(raw_path, dest_img)
                
                # Save YOLO segmentation annotation
                label_filename = raw_path.stem + '.txt'
                dest_label = self.output_dir / 'labels' / split / label_filename
                self.save_yolo_seg_annotation(polygons, dest_label)
                
                # Stats
                total_polygons += len(polygons)
                total_points += sum(len(p) // 2 for p in polygons)
                if polygons:
                    images_with_annotations += 1
            
            stats[split] = {
                'images': len(split_pairs),
                'images_with_annotations': images_with_annotations,
                'total_polygons': total_polygons,
                'avg_polygons_per_image': total_polygons / max(len(split_pairs), 1),
                'avg_points_per_polygon': total_points / max(total_polygons, 1),
            }
            
            print(f"  Images: {len(split_pairs)} ({images_with_annotations} with annotations)")
            print(f"  Total polygons: {total_polygons}")
            print(f"  Avg polygons/image: {stats[split]['avg_polygons_per_image']:.1f}")
            print(f"  Avg points/polygon: {stats[split]['avg_points_per_polygon']:.1f}")
        
        # Create data.yaml
        self._create_data_yaml()
        
        # Print summary
        print("\n" + "=" * 70)
        print("DATASET SUMMARY")
        print("=" * 70)
        for split, s in stats.items():
            print(f"\n{split.upper()}:")
            print(f"  Images: {s['images']} ({s['images_with_annotations']} annotated)")
            print(f"  Polygons: {s['total_polygons']}")
        print(f"\nOutput saved to: {self.output_dir}")
        print("=" * 70)
    
    def _create_data_yaml(self):
        """Create the YOLO data.yaml configuration file for segmentation."""
        data_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': self.CLASSES,
            'nc': len(self.CLASSES),
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"\nCreated data.yaml at: {yaml_path}")
    
    def visualize_sample(self, split: str = 'train', n_samples: int = 3):
        """
        Visualize sample annotations to verify correctness.
        Saves visualization images to output_dir/visualizations/
        
        Args:
            split: 'train' or 'val'
            n_samples: Number of samples to visualize
        """
        vis_dir = self.output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        img_dir = self.output_dir / 'images' / split
        label_dir = self.output_dir / 'labels' / split
        
        images = sorted(img_dir.glob('*.jpg'))[:n_samples]
        
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            overlay = img.copy()
            
            label_path = label_dir / (img_path.stem + '.txt')
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 7:  # class_id + at least 3 points
                            continue
                        
                        # Parse polygon
                        coords = [float(x) for x in parts[1:]]
                        points = []
                        for i in range(0, len(coords), 2):
                            px = int(coords[i] * w)
                            py = int(coords[i + 1] * h)
                            points.append([px, py])
                        
                        pts = np.array(points, np.int32).reshape((-1, 1, 2))
                        
                        # Draw filled polygon with transparency
                        cv2.fillPoly(overlay, [pts], (0, 0, 255))
                        # Draw polygon outline
                        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            
            # Blend overlay
            result = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
            
            # Add text
            cv2.putText(result, f"File: {img_path.name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            vis_path = vis_dir / f"vis_{img_path.name}"
            cv2.imwrite(str(vis_path), result)
            print(f"  Saved visualization: {vis_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare hull defect dataset for YOLO Segmentation training'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to MaVeCoDD dataset root (ry392rp8cj-1 folder)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='datasets/hull_defects_seg',
        help='Output directory for YOLO-seg dataset'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training set ratio (default 0.8)'
    )
    parser.add_argument(
        '--min-area',
        type=int,
        default=150,
        help='Minimum contour area in pixels'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.005,
        help='Polygon approximation epsilon factor'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization of annotations'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible splits'
    )
    
    args = parser.parse_args()
    
    preparer = SegDatasetPreparer(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=1.0 - args.train_ratio,
        min_contour_area=args.min_area,
        epsilon_factor=args.epsilon,
    )
    
    preparer.prepare_dataset(seed=args.seed)
    
    if args.visualize:
        print("\nGenerating visualizations...")
        preparer.visualize_sample('train', n_samples=5)
        preparer.visualize_sample('val', n_samples=3)


if __name__ == '__main__':
    main()
