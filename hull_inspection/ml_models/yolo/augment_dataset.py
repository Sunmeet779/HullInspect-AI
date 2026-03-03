#!/usr/bin/env python3
"""
Advanced Data Augmentation for Hull Defect Detection
Generates multiple augmented versions of each image to expand the dataset.
"""

import os
import cv2
import numpy as np
import random
import shutil
from pathlib import Path
from typing import List, Tuple
import albumentations as A
from tqdm import tqdm


class YOLOAugmentor:
    """Advanced augmentation for YOLO format datasets."""
    
    def __init__(self, dataset_path: str, output_path: str = None, augment_factor: int = 10):
        """
        Args:
            dataset_path: Path to the original dataset (with images/ and labels/ subdirs)
            output_path: Path for augmented dataset (default: dataset_path + '_augmented')
            augment_factor: Number of augmented versions per original image
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path) if output_path else self.dataset_path.parent / f"{self.dataset_path.name}_augmented"
        self.augment_factor = augment_factor
        
        # Define augmentation pipelines
        self.augmentations = self._create_augmentation_pipelines()
        
    def _create_augmentation_pipelines(self) -> List[A.Compose]:
        """Create diverse augmentation pipelines."""
        
        # Pipeline 1: Geometric transforms
        geometric = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=45, p=0.7, border_mode=cv2.BORDER_REFLECT),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-30, 30),
                shear=(-10, 10),
                p=0.5
            ),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
        # Pipeline 2: Color/lighting transforms
        color = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=30, p=0.7),
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
        # Pipeline 3: Weather/environmental simulation
        weather = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=100, p=1),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=1),
            ], p=0.7),
            A.RandomBrightnessContrast(p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
        # Pipeline 4: Noise and blur
        noise = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1),
            ], p=0.6),
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1),
                A.GaussianBlur(blur_limit=(3, 7), p=1),
                A.MedianBlur(blur_limit=5, p=1),
            ], p=0.4),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
        # Pipeline 5: Heavy augmentation (combine multiple)
        heavy = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_REFLECT),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 30), p=1),
                A.GaussianBlur(blur_limit=5, p=1),
            ], p=0.3),
            A.HueSaturationValue(p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
        # Pipeline 6: Perspective and distortion
        perspective = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Perspective(scale=(0.05, 0.1), p=0.5),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
            A.RandomBrightnessContrast(p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
        # Pipeline 7: Cropping and padding
        crop = A.Compose([
            A.RandomCrop(height=480, width=480, p=0.5),
            A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_REFLECT, p=1),
            A.Resize(640, 640),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
        # Pipeline 8: Underwater/marine simulation
        marine = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=30, p=0.7),  # Bluish tint
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.1), contrast_limit=0.2, p=0.7),
            A.GaussNoise(var_limit=(5, 25), p=0.4),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
        # Pipeline 9: Texture enhancement
        texture = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.CLAHE(clip_limit=4.0, p=0.6),
            A.Sharpen(alpha=(0.2, 0.5), p=0.5),
            A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.3),
            A.RandomBrightnessContrast(p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
        # Pipeline 10: Mixed strong augmentation
        mixed = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
                A.RGBShift(p=1),
            ], p=0.8),
            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(p=1),
                A.MotionBlur(p=1),
            ], p=0.4),
            A.OneOf([
                A.Rotate(limit=45, p=1),
                A.Affine(scale=(0.8, 1.2), p=1),
                A.Perspective(p=1),
            ], p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        
        return [geometric, color, weather, noise, heavy, perspective, crop, marine, texture, mixed]
    
    def read_yolo_labels(self, label_path: Path) -> Tuple[List[List[float]], List[int]]:
        """Read YOLO format labels."""
        bboxes = []
        class_labels = []
        
        if not label_path.exists():
            return bboxes, class_labels
            
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    # Clamp values to valid range
                    x_center = max(0.001, min(0.999, x_center))
                    y_center = max(0.001, min(0.999, y_center))
                    width = max(0.001, min(0.999, width))
                    height = max(0.001, min(0.999, height))
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
        
        return bboxes, class_labels
    
    def write_yolo_labels(self, label_path: Path, bboxes: List[List[float]], class_labels: List[int]):
        """Write YOLO format labels."""
        with open(label_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                x_center, y_center, width, height = bbox
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def augment_image(self, image: np.ndarray, bboxes: List[List[float]], 
                      class_labels: List[int], pipeline_idx: int) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """Apply augmentation to a single image."""
        if not bboxes:
            # If no bboxes, just apply augmentation without bbox params
            simple_aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Rotate(limit=30, p=0.5),
            ])
            augmented = simple_aug(image=image)
            return augmented['image'], [], []
        
        pipeline = self.augmentations[pipeline_idx % len(self.augmentations)]
        
        try:
            augmented = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
            return augmented['image'], augmented['bboxes'], augmented['class_labels']
        except Exception as e:
            # Fallback to original if augmentation fails
            return image, bboxes, class_labels
    
    def process_split(self, split: str):
        """Process train or val split."""
        input_images_dir = self.dataset_path / 'images' / split
        input_labels_dir = self.dataset_path / 'labels' / split
        output_images_dir = self.output_path / 'images' / split
        output_labels_dir = self.output_path / 'labels' / split
        
        # Create output directories
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = list(input_images_dir.glob('*.jpg')) + list(input_images_dir.glob('*.png'))
        
        print(f"\nProcessing {split} split: {len(image_files)} images")
        print(f"Generating {self.augment_factor} augmented versions per image...")
        
        total_images = 0
        total_annotations = 0
        
        for img_path in tqdm(image_files, desc=f"Augmenting {split}"):
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Read labels
            label_path = input_labels_dir / f"{img_path.stem}.txt"
            bboxes, class_labels = self.read_yolo_labels(label_path)
            
            # Save original image
            original_name = f"{img_path.stem}_orig"
            cv2.imwrite(str(output_images_dir / f"{original_name}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            self.write_yolo_labels(output_labels_dir / f"{original_name}.txt", bboxes, class_labels)
            total_images += 1
            total_annotations += len(bboxes)
            
            # Generate augmented versions
            for i in range(self.augment_factor):
                try:
                    aug_image, aug_bboxes, aug_class_labels = self.augment_image(
                        image.copy(), bboxes.copy(), class_labels.copy(), i
                    )
                    
                    # Only save if we still have valid bboxes (or if original had none)
                    if aug_bboxes or not bboxes:
                        aug_name = f"{img_path.stem}_aug{i:02d}"
                        cv2.imwrite(
                            str(output_images_dir / f"{aug_name}.jpg"),
                            cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                        )
                        self.write_yolo_labels(
                            output_labels_dir / f"{aug_name}.txt",
                            aug_bboxes,
                            aug_class_labels
                        )
                        total_images += 1
                        total_annotations += len(aug_bboxes)
                except Exception as e:
                    continue
        
        return total_images, total_annotations
    
    def run(self):
        """Run the augmentation pipeline."""
        print("=" * 60)
        print("YOLO Dataset Augmentation")
        print("=" * 60)
        print(f"Input dataset: {self.dataset_path}")
        print(f"Output dataset: {self.output_path}")
        print(f"Augmentation factor: {self.augment_factor}x per image")
        print("=" * 60)
        
        # Process train split
        train_images, train_annotations = self.process_split('train')
        
        # Process val split (with fewer augmentations)
        original_factor = self.augment_factor
        self.augment_factor = 3  # Less augmentation for validation
        val_images, val_annotations = self.process_split('val')
        self.augment_factor = original_factor
        
        # Create data.yaml
        self._create_data_yaml()
        
        print("\n" + "=" * 60)
        print("AUGMENTATION COMPLETE!")
        print("=" * 60)
        print(f"Training images: {train_images} (annotations: {train_annotations})")
        print(f"Validation images: {val_images} (annotations: {val_annotations})")
        print(f"Total images: {train_images + val_images}")
        print(f"Total annotations: {train_annotations + val_annotations}")
        print(f"\nAugmented dataset saved to: {self.output_path}")
        print("=" * 60)
        
        return train_images, val_images
    
    def _create_data_yaml(self):
        """Create data.yaml for the augmented dataset."""
        yaml_content = f"""# Augmented Hull Defects Dataset
# Auto-generated by augment_dataset.py

path: {self.output_path}
train: images/train
val: images/val

# Classes
names:
  0: corrosion_severe
  1: corrosion_moderate
  2: corrosion_mild
  3: paint_damage
  4: fouling
  5: crack
  6: dent
  7: surface_defect

nc: 8
"""
        yaml_path = self.output_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"\nCreated: {yaml_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Augment YOLO dataset for hull defect detection')
    parser.add_argument('--input', type=str, 
                       default='/Users/sunmeetkohli/inspection report/research-project/hull_inspection/datasets/hull_defects',
                       help='Path to input dataset')
    parser.add_argument('--output', type=str, default=None,
                       help='Path for output dataset (default: input_augmented)')
    parser.add_argument('--factor', type=int, default=10,
                       help='Number of augmented versions per image (default: 10)')
    
    args = parser.parse_args()
    
    # Check if albumentations is installed
    try:
        import albumentations
    except ImportError:
        print("Installing albumentations...")
        os.system('pip install albumentations')
        import albumentations
    
    augmentor = YOLOAugmentor(
        dataset_path=args.input,
        output_path=args.output,
        augment_factor=args.factor
    )
    augmentor.run()


if __name__ == '__main__':
    main()
