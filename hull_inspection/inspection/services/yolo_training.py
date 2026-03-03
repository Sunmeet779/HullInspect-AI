"""
YOLO Training Script for Hull Defect Detection

This script handles:
- Dataset preparation and validation
- Model training with proper configuration
- Training monitoring and result checking
- Model evaluation and metrics
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import json


class YOLOTrainer:
    """Trainer class for YOLO hull defect detection model."""
    
    def __init__(
        self,
        data_yaml_path: str,
        model_size: str = "yolov8n",  # n, s, m, l, x
        project_dir: str = "runs/hull_detection",
        experiment_name: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            data_yaml_path: Path to dataset YAML file
            model_size: YOLO model size (yolov8n, yolov8s, etc.)
            project_dir: Directory to save training runs
            experiment_name: Name for this experiment
        """
        self.data_yaml_path = data_yaml_path
        self.model_size = model_size
        self.project_dir = project_dir
        self.experiment_name = experiment_name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model = None
        self.results = None
        
    def check_environment(self) -> Dict[str, bool]:
        """Check if training environment is properly set up."""
        checks = {
            "ultralytics_installed": False,
            "torch_installed": False,
            "gpu_available": False,
            "mps_available": False,
            "dataset_exists": False,
            "dataset_valid": False,
        }
        
        # Check ultralytics
        try:
            from ultralytics import YOLO
            checks["ultralytics_installed"] = True
        except ImportError:
            print("❌ ultralytics not installed. Run: pip install ultralytics")
        
        # Check torch
        try:
            import torch
            checks["torch_installed"] = True
            checks["gpu_available"] = torch.cuda.is_available()
            checks["mps_available"] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            
            if checks["gpu_available"]:
                print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            elif checks["mps_available"]:
                print("✅ Apple MPS (Metal) available")
            else:
                print("⚠️ No GPU detected, training will use CPU (slower)")
                
        except ImportError:
            print("❌ PyTorch not installed. Run: pip install torch")
        
        # Check dataset
        if os.path.exists(self.data_yaml_path):
            checks["dataset_exists"] = True
            checks["dataset_valid"] = self._validate_dataset()
        else:
            print(f"❌ Dataset YAML not found: {self.data_yaml_path}")
        
        return checks
    
    def _validate_dataset(self) -> bool:
        """Validate dataset structure."""
        import yaml
        
        try:
            with open(self.data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            required_keys = ['train', 'val', 'names']
            for key in required_keys:
                if key not in data_config:
                    print(f"❌ Missing '{key}' in dataset YAML")
                    return False
            
            # Check if paths exist
            base_path = Path(self.data_yaml_path).parent
            
            train_path = base_path / data_config['train']
            val_path = base_path / data_config['val']
            
            if not train_path.exists():
                print(f"❌ Training images not found: {train_path}")
                return False
            
            if not val_path.exists():
                print(f"❌ Validation images not found: {val_path}")
                return False
            
            # Count images
            train_images = list(train_path.glob("*.jpg")) + list(train_path.glob("*.png"))
            val_images = list(val_path.glob("*.jpg")) + list(val_path.glob("*.png"))
            
            print(f"✅ Dataset validated:")
            print(f"   Training images: {len(train_images)}")
            print(f"   Validation images: {len(val_images)}")
            print(f"   Classes: {data_config['names']}")
            
            return len(train_images) > 0 and len(val_images) > 0
            
        except Exception as e:
            print(f"❌ Error validating dataset: {e}")
            return False
    
    def train(
        self,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        patience: int = 20,
        save_period: int = 10,
        resume: bool = False,
        pretrained: bool = True,
        **kwargs
    ) -> Dict:
        """
        Train the YOLO model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size (reduce if OOM)
            img_size: Input image size
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            resume: Resume from last checkpoint
            pretrained: Use pretrained weights
            **kwargs: Additional training arguments
            
        Returns:
            Training results dictionary
        """
        from ultralytics import YOLO
        
        print("\n" + "=" * 60)
        print("STARTING YOLO TRAINING")
        print("=" * 60)
        
        # Load model
        if resume and os.path.exists(f"{self.project_dir}/{self.experiment_name}/weights/last.pt"):
            print("📥 Resuming from last checkpoint...")
            self.model = YOLO(f"{self.project_dir}/{self.experiment_name}/weights/last.pt")
        else:
            print(f"📥 Loading {self.model_size} {'with pretrained weights' if pretrained else 'from scratch'}...")
            self.model = YOLO(f"{self.model_size}.pt" if pretrained else f"{self.model_size}.yaml")
        
        # Start training
        print(f"\n🚀 Training for {epochs} epochs...")
        print(f"   Batch size: {batch_size}")
        print(f"   Image size: {img_size}")
        print(f"   Project: {self.project_dir}/{self.experiment_name}")
        print("\n")
        
        self.results = self.model.train(
            data=self.data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            patience=patience,
            save_period=save_period,
            project=self.project_dir,
            name=self.experiment_name,
            exist_ok=True,
            verbose=True,
            plots=True,  # Generate training plots
            save=True,   # Save checkpoints
            **kwargs
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        
        return self._get_training_summary()
    
    def _get_training_summary(self) -> Dict:
        """Get summary of training results."""
        results_dir = Path(self.project_dir) / self.experiment_name
        
        summary = {
            "experiment": self.experiment_name,
            "results_dir": str(results_dir),
            "weights": {},
            "metrics": {},
            "plots": []
        }
        
        # Check for weights
        weights_dir = results_dir / "weights"
        if weights_dir.exists():
            if (weights_dir / "best.pt").exists():
                summary["weights"]["best"] = str(weights_dir / "best.pt")
            if (weights_dir / "last.pt").exists():
                summary["weights"]["last"] = str(weights_dir / "last.pt")
        
        # Check for results CSV
        results_csv = results_dir / "results.csv"
        if results_csv.exists():
            summary["metrics"] = self._parse_results_csv(results_csv)
        
        # Check for plots
        plot_files = ["results.png", "confusion_matrix.png", "F1_curve.png", 
                      "P_curve.png", "R_curve.png", "PR_curve.png"]
        for plot in plot_files:
            plot_path = results_dir / plot
            if plot_path.exists():
                summary["plots"].append(str(plot_path))
        
        return summary
    
    def _parse_results_csv(self, csv_path: Path) -> Dict:
        """Parse training results CSV file."""
        import csv
        
        metrics = {
            "epochs_completed": 0,
            "final_metrics": {},
            "best_metrics": {}
        }
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if rows:
                    metrics["epochs_completed"] = len(rows)
                    
                    # Get final epoch metrics
                    last_row = rows[-1]
                    metrics["final_metrics"] = {
                        k.strip(): float(v) if v.replace('.', '').replace('-', '').isdigit() else v
                        for k, v in last_row.items()
                    }
                    
                    # Find best mAP50
                    best_map50 = 0
                    best_row = None
                    for row in rows:
                        map50_key = [k for k in row.keys() if 'mAP50' in k and 'mAP50-95' not in k]
                        if map50_key:
                            try:
                                map50 = float(row[map50_key[0]])
                                if map50 > best_map50:
                                    best_map50 = map50
                                    best_row = row
                            except (ValueError, KeyError):
                                pass
                    
                    if best_row:
                        metrics["best_metrics"] = {
                            k.strip(): float(v) if v.replace('.', '').replace('-', '').isdigit() else v
                            for k, v in best_row.items()
                        }
                        
        except Exception as e:
            print(f"Warning: Could not parse results CSV: {e}")
        
        return metrics
    
    def evaluate(self, weights_path: Optional[str] = None) -> Dict:
        """
        Evaluate the trained model.
        
        Args:
            weights_path: Path to model weights (uses best.pt if None)
            
        Returns:
            Evaluation metrics
        """
        from ultralytics import YOLO
        
        if weights_path is None:
            weights_path = f"{self.project_dir}/{self.experiment_name}/weights/best.pt"
        
        if not os.path.exists(weights_path):
            print(f"❌ Weights not found: {weights_path}")
            return {}
        
        print(f"\n📊 Evaluating model: {weights_path}")
        
        model = YOLO(weights_path)
        metrics = model.val(data=self.data_yaml_path, verbose=True)
        
        results = {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "per_class": {}
        }
        
        # Per-class metrics
        if hasattr(metrics.box, 'ap_class_index'):
            for i, class_idx in enumerate(metrics.box.ap_class_index):
                class_name = model.names[int(class_idx)]
                results["per_class"][class_name] = {
                    "precision": float(metrics.box.p[i]),
                    "recall": float(metrics.box.r[i]),
                    "ap50": float(metrics.box.ap50[i]),
                    "ap": float(metrics.box.ap[i])
                }
        
        return results
    
    def check_training_progress(self) -> Dict:
        """Check if training is in progress and get current status."""
        results_dir = Path(self.project_dir) / self.experiment_name
        
        status = {
            "training_started": False,
            "training_completed": False,
            "current_epoch": 0,
            "has_checkpoints": False,
            "last_update": None
        }
        
        if not results_dir.exists():
            return status
        
        status["training_started"] = True
        
        # Check for results CSV
        results_csv = results_dir / "results.csv"
        if results_csv.exists():
            import csv
            with open(results_csv, 'r') as f:
                rows = list(csv.reader(f))
                status["current_epoch"] = len(rows) - 1  # Subtract header
            
            status["last_update"] = datetime.fromtimestamp(
                results_csv.stat().st_mtime
            ).strftime("%Y-%m-%d %H:%M:%S")
        
        # Check for weights
        weights_dir = results_dir / "weights"
        if weights_dir.exists():
            status["has_checkpoints"] = (weights_dir / "last.pt").exists()
            status["training_completed"] = (weights_dir / "best.pt").exists()
        
        return status


def create_sample_dataset_yaml(output_path: str, dataset_root: str):
    """Create a sample dataset YAML file for hull defect detection."""
    
    yaml_content = f"""# Hull Defect Detection Dataset
# Created: {datetime.now().strftime('%Y-%m-%d')}

path: {dataset_root}  # dataset root dir
train: images/train   # train images (relative to path)
val: images/val       # val images (relative to path)
test: images/test     # test images (optional)

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

# Dataset info
nc: 8  # number of classes
"""
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Created dataset YAML: {output_path}")
    return output_path


def verify_training_results(runs_dir: str = "runs/hull_detection") -> None:
    """
    Verify and display all training results.
    
    Args:
        runs_dir: Directory containing training runs
    """
    runs_path = Path(runs_dir)
    
    if not runs_path.exists():
        print(f"❌ No training runs found at: {runs_dir}")
        print("\n💡 To start training, run:")
        print("   trainer = YOLOTrainer('path/to/data.yaml')")
        print("   trainer.train(epochs=100)")
        return
    
    print("\n" + "=" * 60)
    print("YOLO TRAINING RESULTS VERIFICATION")
    print("=" * 60)
    
    experiments = list(runs_path.iterdir())
    if not experiments:
        print("No experiments found.")
        return
    
    for exp_dir in sorted(experiments):
        if not exp_dir.is_dir():
            continue
            
        print(f"\n📁 Experiment: {exp_dir.name}")
        print("-" * 40)
        
        # Check weights
        weights_dir = exp_dir / "weights"
        if weights_dir.exists():
            best_pt = weights_dir / "best.pt"
            last_pt = weights_dir / "last.pt"
            
            if best_pt.exists():
                size_mb = best_pt.stat().st_size / (1024 * 1024)
                print(f"  ✅ best.pt: {size_mb:.1f} MB")
            else:
                print("  ❌ best.pt: Not found (training incomplete?)")
            
            if last_pt.exists():
                size_mb = last_pt.stat().st_size / (1024 * 1024)
                print(f"  ✅ last.pt: {size_mb:.1f} MB")
        else:
            print("  ❌ No weights directory found")
        
        # Check results CSV
        results_csv = exp_dir / "results.csv"
        if results_csv.exists():
            import csv
            with open(results_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if rows:
                    print(f"  📊 Epochs completed: {len(rows)}")
                    
                    # Get final metrics
                    last = rows[-1]
                    metrics_to_show = [
                        ('metrics/mAP50(B)', 'mAP50'),
                        ('metrics/mAP50-95(B)', 'mAP50-95'),
                        ('metrics/precision(B)', 'Precision'),
                        ('metrics/recall(B)', 'Recall')
                    ]
                    
                    print("  📈 Final Metrics:")
                    for csv_key, display_name in metrics_to_show:
                        # Handle different CSV column name formats
                        value = None
                        for key in last.keys():
                            if csv_key.replace('(B)', '') in key or display_name.lower() in key.lower():
                                try:
                                    value = float(last[key])
                                    break
                                except (ValueError, TypeError):
                                    pass
                        
                        if value is not None:
                            print(f"      {display_name}: {value:.4f}")
        else:
            print("  ⚠️ No results.csv found (training may be in progress)")
        
        # Check for plots
        plots = list(exp_dir.glob("*.png"))
        if plots:
            print(f"  🖼️ Generated plots: {len(plots)}")
            for plot in plots[:5]:
                print(f"      - {plot.name}")


# Main entry point for testing
if __name__ == "__main__":
    print("YOLO Training Verification Script")
    print("=" * 40)
    
    # Check for existing training runs
    verify_training_results()
    
    # Example usage
    print("\n" + "=" * 60)
    print("EXAMPLE USAGE")
    print("=" * 60)
    print("""
# 1. First, create your dataset YAML:
create_sample_dataset_yaml('hull_dataset.yaml', '/path/to/dataset')

# 2. Initialize trainer:
trainer = YOLOTrainer(
    data_yaml_path='hull_dataset.yaml',
    model_size='yolov8n',  # Use 'yolov8s' or 'yolov8m' for better accuracy
    experiment_name='hull_v1'
)

# 3. Check environment:
checks = trainer.check_environment()

# 4. Start training:
if all(checks.values()):
    results = trainer.train(
        epochs=100,
        batch_size=16,
        img_size=640,
        patience=20
    )

# 5. Evaluate:
metrics = trainer.evaluate()
print(f"mAP50: {metrics['mAP50']:.4f}")
print(f"mAP50-95: {metrics['mAP50-95']:.4f}")

# 6. Check training progress anytime:
status = trainer.check_training_progress()
print(f"Current epoch: {status['current_epoch']}")
""")
