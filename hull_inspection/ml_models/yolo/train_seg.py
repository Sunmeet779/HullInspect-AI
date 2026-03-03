"""
YOLO11-Seg Training Script for Hull Corrosion Segmentation

This script handles:
- Dataset validation for segmentation task
- Model training with YOLO11-seg (instance segmentation)
- Augmentation optimized for corrosion on Mac M1
- Training monitoring, evaluation, and export

Key differences from the bounding-box version:
- Uses YOLO11n-seg model instead of YOLOv8n detection model
- Single "corrosion" class (no heuristic sub-classes)
- Segmentation loss (mask + box + cls + dfl)
- Polygon-based annotations
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import json


class YOLOSegTrainer:
    """Trainer class for YOLO11-Seg hull corrosion segmentation model."""
    
    def __init__(
        self,
        data_yaml_path: str,
        model_size: str = "yolo11n-seg",  # n, s, m, l, x  (all with -seg suffix)
        project_dir: str = "runs/segment",
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize the segmentation trainer.
        
        Args:
            data_yaml_path: Path to dataset YAML file
            model_size: YOLO model size (yolo11n-seg, yolo11s-seg, etc.)
            project_dir: Directory to save training runs
            experiment_name: Name for this experiment
        """
        self.data_yaml_path = data_yaml_path
        self.model_size = model_size
        self.project_dir = project_dir
        self.experiment_name = experiment_name or f"seg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model = None
        self.results = None
        
    def check_environment(self) -> Dict[str, bool]:
        """Check if training environment is properly set up."""
        checks = {
            "ultralytics_installed": False,
            "torch_installed": False,
            "mps_available": False,
            "gpu_available": False,
            "dataset_exists": False,
            "dataset_valid": False,
        }
        
        try:
            from ultralytics import YOLO
            import ultralytics
            checks["ultralytics_installed"] = True
            print(f"✅ Ultralytics version: {ultralytics.__version__}")
        except ImportError:
            print("❌ ultralytics not installed. Run: pip install ultralytics>=8.1.0")
        
        try:
            import torch
            checks["torch_installed"] = True
            print(f"✅ PyTorch version: {torch.__version__}")
            
            checks["gpu_available"] = torch.cuda.is_available()
            checks["mps_available"] = (
                hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            )
            
            if checks["mps_available"]:
                print("✅ Apple MPS (Metal) available — will use GPU acceleration")
            elif checks["gpu_available"]:
                print(f"✅ CUDA GPU available: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️ No GPU detected, training will use CPU (slower)")
                
        except ImportError:
            print("❌ PyTorch not installed")
        
        if os.path.exists(self.data_yaml_path):
            checks["dataset_exists"] = True
            checks["dataset_valid"] = self._validate_dataset()
        else:
            print(f"❌ Dataset YAML not found: {self.data_yaml_path}")
        
        return checks
    
    def _validate_dataset(self) -> bool:
        """Validate dataset structure for segmentation."""
        import yaml
        
        try:
            with open(self.data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            for key in ['train', 'val', 'names']:
                if key not in data_config:
                    print(f"❌ Missing '{key}' in dataset YAML")
                    return False
            
            base_path = Path(self.data_yaml_path).parent
            train_img_path = base_path / data_config['train']
            val_img_path = base_path / data_config['val']
            
            # Check images exist
            train_images = list(train_img_path.glob("*.jpg")) + list(train_img_path.glob("*.png"))
            val_images = list(val_img_path.glob("*.jpg")) + list(val_img_path.glob("*.png"))
            
            if not train_images:
                print(f"❌ No training images in: {train_img_path}")
                return False
            if not val_images:
                print(f"❌ No validation images in: {val_img_path}")
                return False
            
            # Check labels exist and contain polygon format
            train_label_path = base_path / data_config['train'].replace('images', 'labels')
            sample_labels = list(Path(train_label_path).glob("*.txt"))[:3]
            
            polygon_format = True
            for label_file in sample_labels:
                with open(label_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 7:  # class + at least 3 (x,y) pairs
                            polygon_format = False
                            break
            
            print(f"✅ Dataset validated (segmentation format: {'YES' if polygon_format else 'NO - bbox?'}):")
            print(f"   Training images: {len(train_images)}")
            print(f"   Validation images: {len(val_images)}")
            print(f"   Classes: {data_config['names']}")
            
            if not polygon_format:
                print("⚠️  Labels may not be in polygon format!")
                print("   Run prepare_dataset_seg.py to convert masks → polygons")
            
            return True
            
        except Exception as e:
            print(f"❌ Error validating dataset: {e}")
            return False
    
    def train(
        self,
        epochs: int = 150,
        batch_size: int = 8,
        img_size: int = 640,
        patience: int = 30,
        save_period: int = 10,
        resume: bool = False,
        pretrained: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Train the YOLO11-Seg model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size (8 recommended for M1 Mac, 16 for GPU)
            img_size: Input image size (640 default)
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            resume: Resume from last checkpoint
            pretrained: Use pretrained COCO-seg weights
            device: Device ('mps', 'cuda', 'cpu', or None for auto)
            **kwargs: Additional YOLO training arguments
            
        Returns:
            Dictionary with training results
        """
        from ultralytics import YOLO
        import torch
        
        # Auto-detect device
        if device is None:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = '0'
            else:
                device = 'cpu'
        
        print(f"\n{'='*70}")
        print(f"YOLO11-Seg Training Configuration")
        print(f"{'='*70}")
        print(f"  Model: {self.model_size}")
        print(f"  Device: {device}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {img_size}")
        print(f"  Data: {self.data_yaml_path}")
        print(f"  Project: {self.project_dir}")
        print(f"  Experiment: {self.experiment_name}")
        print(f"{'='*70}\n")
        
        # Load model
        if resume:
            last_path = Path(self.project_dir) / self.experiment_name / 'weights' / 'last.pt'
            if last_path.exists():
                print(f"Resuming from: {last_path}")
                self.model = YOLO(str(last_path))
            else:
                print(f"No checkpoint found at {last_path}, starting fresh")
                self.model = YOLO(f'{self.model_size}.pt')
        else:
            self.model = YOLO(f'{self.model_size}.pt')
        
        # Training arguments optimized for corrosion segmentation on M1 Mac
        train_args = {
            'data': self.data_yaml_path,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'patience': patience,
            'save_period': save_period,
            'device': device,
            'project': self.project_dir,
            'name': self.experiment_name,
            'exist_ok': True,
            'pretrained': pretrained,
            'verbose': True,
            
            # Optimizer — AdamW works better for small datasets
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'weight_decay': 0.0005,
            'warmup_epochs': 5.0,
            'warmup_momentum': 0.8,
            
            # Loss weights — boost mask loss for better segmentation
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            
            # Augmentation — optimized for corrosion images
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 15.0,      # Some rotation helps for corrosion
            'translate': 0.15,
            'scale': 0.5,
            'shear': 5.0,         # Mild shear
            'perspective': 0.0,
            'flipud': 0.3,        # Corrosion can appear in any orientation
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.15,
            'copy_paste': 0.2,    # Copy-paste good for instance segmentation
            
            # Misc
            'workers': 4,
            'seed': 42,
            'deterministic': True,
            'single_cls': True,   # All annotations are class 0 (corrosion)
            'plots': True,
        }
        
        # Override with user kwargs
        train_args.update(kwargs)
        
        # Train
        try:
            self.results = self.model.train(**train_args)
            
            # Save training info
            info = {
                'model': self.model_size,
                'epochs': epochs,
                'batch_size': batch_size,
                'img_size': img_size,
                'device': device,
                'data_yaml': self.data_yaml_path,
                'completed_at': datetime.now().isoformat(),
            }
            
            info_path = Path(self.project_dir) / self.experiment_name / 'training_info.json'
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
            
            print(f"\n✅ Training complete!")
            print(f"   Best weights: {self.project_dir}/{self.experiment_name}/weights/best.pt")
            
            return {'success': True, 'results': str(self.results)}
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def validate(self, model_path: Optional[str] = None) -> Dict:
        """Validate the model on the validation set."""
        from ultralytics import YOLO
        
        if model_path:
            model = YOLO(model_path)
        elif self.model:
            model = self.model
        else:
            best = Path(self.project_dir) / self.experiment_name / 'weights' / 'best.pt'
            if best.exists():
                model = YOLO(str(best))
            else:
                print("No model found for validation")
                return {}
        
        results = model.val(data=self.data_yaml_path)
        
        metrics = {
            'mAP50_mask': float(results.seg.map50) if hasattr(results, 'seg') else None,
            'mAP50_95_mask': float(results.seg.map) if hasattr(results, 'seg') else None,
            'mAP50_box': float(results.box.map50),
            'mAP50_95_box': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
        }
        
        print(f"\nValidation Results:")
        for k, v in metrics.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")
        
        return metrics
    
    def export(
        self,
        model_path: Optional[str] = None,
        formats: Optional[list] = None,
    ):
        """Export the trained model to various formats for deployment."""
        from ultralytics import YOLO
        
        if formats is None:
            formats = ['coreml']  # CoreML optimal for M1 Mac
        
        if model_path:
            model = YOLO(model_path)
        else:
            best = Path(self.project_dir) / self.experiment_name / 'weights' / 'best.pt'
            model = YOLO(str(best))
        
        for fmt in formats:
            print(f"\nExporting to {fmt}...")
            model.export(format=fmt, imgsz=640)
            print(f"✅ Exported to {fmt}")


def main():
    """Main entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLO11-Seg for hull corrosion')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolo11n-seg', help='Model size')
    parser.add_argument('--epochs', type=int, default=150, help='Training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default=None, help='Device (mps/cuda/cpu)')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--validate-only', action='store_true', help='Only validate')
    parser.add_argument('--export', action='store_true', help='Export after training')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    
    args = parser.parse_args()
    
    trainer = YOLOSegTrainer(
        data_yaml_path=args.data,
        model_size=args.model,
        experiment_name=args.name,
    )
    
    # Check environment
    checks = trainer.check_environment()
    
    if not all([checks['ultralytics_installed'], checks['torch_installed'], checks['dataset_valid']]):
        print("\n❌ Environment check failed. Fix issues above before training.")
        sys.exit(1)
    
    if args.validate_only:
        trainer.validate()
    else:
        result = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.imgsz,
            device=args.device,
            resume=args.resume,
        )
        
        if result.get('success') and args.export:
            trainer.export()


if __name__ == '__main__':
    main()
