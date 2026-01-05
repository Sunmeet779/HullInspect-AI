"""
Dataset Preparation Script for YOLO Training

This script converts the MaVeCoDD dataset (Marine Vessel Hull Corrosion in Dry-Dock)
from its labeled mask format to YOLO bounding box format.

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
    └── hull_defects/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   ├── val/
        │   └── test/
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


class DatasetPreparer:
    """
    Prepares hull defect dataset for YOLO training.
    Converts labeled mask images to YOLO bounding box format.
    """
    
    # Class definitions for hull defects
    CLASSES = {
        0: 'corrosion_severe',
        1: 'corrosion_moderate',
        2: 'corrosion_mild',
        3: 'paint_damage',
        4: 'fouling',
        5: 'crack',
        6: 'dent',
        7: 'surface_defect',
    }
    
    # For the MaVeCoDD dataset, we primarily have corrosion
    # We'll classify based on region size and intensity
    CORROSION_THRESHOLDS = {
        'severe': 0.05,     # Area > 5% of image
        'moderate': 0.02,   # Area > 2% of image
        'mild': 0.0,        # Any smaller area
    }
    
    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1
    ):
        """
        Initialize the dataset preparer.
        
        Args:
            source_dir: Path to the MaVeCoDD dataset root
            output_dir: Path to output YOLO-formatted dataset
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Create output directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create the YOLO dataset directory structure."""
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    def get_image_pairs(self) -> List[Tuple[Path, Path]]:
        """Get pairs of (raw_image, labeled_image) from the dataset."""
        pairs = []
        
        for resolution in ['HiRes', 'LoRes']:
            raw_dir = self.source_dir / resolution / 'raw'
            labeled_dir = self.source_dir / resolution / 'labeled'
            
            if not raw_dir.exists():
                continue
            
            for raw_path in raw_dir.glob('*.jpg'):
                # Find corresponding labeled image
                base_name = raw_path.stem
                labeled_path = labeled_dir / f"{base_name}_label.jpg"
                
                if labeled_path.exists():
                    pairs.append((raw_path, labeled_path))
                else:
                    print(f"Warning: No label found for {raw_path.name}")
        
        return pairs
    
    def mask_to_bboxes(
        self, 
        label_image: np.ndarray,
        min_area: int = 100
    ) -> List[Dict]:
        """
        Convert a labeled mask image to bounding boxes.
        
        The MaVeCoDD dataset uses colored regions to indicate corrosion.
        We detect these regions and create bounding boxes.
        
        Args:
            label_image: BGR label image
            min_area: Minimum contour area to consider
            
        Returns:
            List of bounding box dictionaries
        """
        # Convert to grayscale
        gray = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY)
        
        # Also check for colored regions (non-gray)
        hsv = cv2.cvtColor(label_image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        # Combine: regions that are either bright in grayscale or have saturation
        mask = np.logical_or(gray > 20, saturation > 30).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        img_height, img_width = label_image.shape[:2]
        img_area = img_height * img_width
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < min_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate relative area for severity classification
            relative_area = area / img_area
            
            # Determine class based on size and color
            class_id, class_name = self._classify_defect(
                label_image[y:y+h, x:x+w],
                relative_area
            )
            
            # Convert to YOLO format (normalized center x, y, width, height)
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            norm_width = w / img_width
            norm_height = h / img_height
            
            bboxes.append({
                'class_id': class_id,
                'class_name': class_name,
                'x_center': x_center,
                'y_center': y_center,
                'width': norm_width,
                'height': norm_height,
                'area': relative_area
            })
        
        return bboxes
    
    def _classify_defect(
        self, 
        region: np.ndarray, 
        relative_area: float
    ) -> Tuple[int, str]:
        """
        Classify a defect based on its appearance and size.
        
        Args:
            region: Image region of the defect
            relative_area: Area of defect relative to full image
            
        Returns:
            Tuple of (class_id, class_name)
        """
        # Analyze color to determine defect type
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Calculate mean color values
        mean_hue = np.mean(hsv[:, :, 0])
        mean_sat = np.mean(hsv[:, :, 1])
        mean_val = np.mean(hsv[:, :, 2])
        
        # Classification rules based on color and size
        # Orange/brown tones indicate rust/corrosion
        if 10 <= mean_hue <= 30 and mean_sat > 50:
            # Rust/corrosion - classify by severity based on size
            if relative_area >= self.CORROSION_THRESHOLDS['severe']:
                return 0, 'corrosion_severe'
            elif relative_area >= self.CORROSION_THRESHOLDS['moderate']:
                return 1, 'corrosion_moderate'
            else:
                return 2, 'corrosion_mild'
        
        # Green tones might indicate marine growth/fouling
        elif 35 <= mean_hue <= 85 and mean_sat > 30:
            return 4, 'fouling'
        
        # Gray/dark tones might be surface defects
        elif mean_sat < 30 and mean_val < 100:
            return 7, 'surface_defect'
        
        # Default to corrosion based on size
        else:
            if relative_area >= self.CORROSION_THRESHOLDS['severe']:
                return 0, 'corrosion_severe'
            elif relative_area >= self.CORROSION_THRESHOLDS['moderate']:
                return 1, 'corrosion_moderate'
            else:
                return 2, 'corrosion_mild'
    
    def process_image_pair(
        self, 
        raw_path: Path, 
        label_path: Path
    ) -> Optional[List[Dict]]:
        """
        Process a single image pair and extract bounding boxes.
        
        Args:
            raw_path: Path to raw image
            label_path: Path to labeled image
            
        Returns:
            List of bounding boxes or None if failed
        """
        try:
            label_img = cv2.imread(str(label_path))
            if label_img is None:
                print(f"Error reading label image: {label_path}")
                return None
            
            bboxes = self.mask_to_bboxes(label_img)
            return bboxes
            
        except Exception as e:
            print(f"Error processing {raw_path.name}: {e}")
            return None
    
    def save_yolo_annotation(
        self, 
        bboxes: List[Dict], 
        output_path: Path
    ):
        """
        Save bounding boxes in YOLO format.
        
        YOLO format: class_id x_center y_center width height
        All values are normalized (0-1)
        
        Args:
            bboxes: List of bounding box dictionaries
            output_path: Path to save the annotation file
        """
        with open(output_path, 'w') as f:
            for bbox in bboxes:
                line = f"{bbox['class_id']} {bbox['x_center']:.6f} {bbox['y_center']:.6f} {bbox['width']:.6f} {bbox['height']:.6f}\n"
                f.write(line)
    
    def prepare_dataset(self):
        """
        Main method to prepare the complete dataset.
        """
        print("=" * 60)
        print("Hull Defect Dataset Preparation")
        print("=" * 60)
        
        # Get all image pairs
        pairs = self.get_image_pairs()
        print(f"Found {len(pairs)} image pairs")
        
        if len(pairs) == 0:
            print("No image pairs found. Check your dataset path.")
            return
        
        # Shuffle and split
        random.shuffle(pairs)
        
        n_total = len(pairs)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train:n_train + n_val]
        test_pairs = pairs[n_train + n_val:]
        
        print(f"Split: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
        
        # Process each split
        stats = {'train': {}, 'val': {}, 'test': {}}
        
        for split, split_pairs in [
            ('train', train_pairs),
            ('val', val_pairs),
            ('test', test_pairs)
        ]:
            print(f"\nProcessing {split} split...")
            
            total_boxes = 0
            class_counts = {name: 0 for name in self.CLASSES.values()}
            
            for raw_path, label_path in split_pairs:
                # Process and get bounding boxes
                bboxes = self.process_image_pair(raw_path, label_path)
                
                if bboxes is None:
                    continue
                
                # Copy raw image
                dest_img_path = self.output_dir / 'images' / split / raw_path.name
                shutil.copy(raw_path, dest_img_path)
                
                # Save YOLO annotation
                label_filename = raw_path.stem + '.txt'
                dest_label_path = self.output_dir / 'labels' / split / label_filename
                self.save_yolo_annotation(bboxes, dest_label_path)
                
                # Update stats
                total_boxes += len(bboxes)
                for bbox in bboxes:
                    class_counts[bbox['class_name']] += 1
            
            stats[split] = {
                'images': len(split_pairs),
                'boxes': total_boxes,
                'class_counts': class_counts
            }
            
            print(f"  - Processed {len(split_pairs)} images with {total_boxes} boxes")
        
        # Create data.yaml
        self.create_data_yaml()
        
        # Print summary
        self.print_summary(stats)
        
        print("\nDataset preparation complete!")
        print(f"Output saved to: {self.output_dir}")
    
    def create_data_yaml(self):
        """Create the YOLO data.yaml configuration file."""
        data_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': self.CLASSES,
            'nc': len(self.CLASSES)
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"\nCreated data.yaml at: {yaml_path}")
    
    def print_summary(self, stats: Dict):
        """Print dataset statistics summary."""
        print("\n" + "=" * 60)
        print("Dataset Statistics")
        print("=" * 60)
        
        for split, data in stats.items():
            print(f"\n{split.upper()}:")
            print(f"  Images: {data['images']}")
            print(f"  Total Boxes: {data['boxes']}")
            print("  Class Distribution:")
            for class_name, count in data['class_counts'].items():
                if count > 0:
                    print(f"    - {class_name}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Prepare hull defect dataset for YOLO training')
    parser.add_argument(
        '--source', 
        type=str, 
        required=True,
        help='Path to MaVeCoDD dataset root (ry392rp8cj-1 folder)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='datasets/hull_defects',
        help='Output directory for YOLO dataset'
    )
    parser.add_argument(
        '--train-ratio', 
        type=float, 
        default=0.7,
        help='Training set ratio'
    )
    parser.add_argument(
        '--val-ratio', 
        type=float, 
        default=0.2,
        help='Validation set ratio'
    )
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1 - args.train_ratio - args.val_ratio
    )
    
    preparer.prepare_dataset()


if __name__ == '__main__':
    main()
