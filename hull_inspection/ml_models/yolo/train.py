"""
YOLO Training Script for Hull Defect Detection

This script trains a YOLOv8 model on the hull defect dataset.
It supports training from scratch or fine-tuning from pretrained weights.

Usage:
    python train.py --data datasets/hull_defects/data.yaml --epochs 100
"""
import os
import argparse
from pathlib import Path
from datetime import datetime
import yaml


def train_yolo(
    data_yaml: str,
    model_size: str = 'n',
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    pretrained: bool = True,
    project: str = 'runs/detect',
    name: str = None,
    device: str = '',
    workers: int = 8,
    patience: int = 50,
    resume: bool = False,
    amp: bool = True
):
    """
    Train YOLOv8 model on hull defect dataset.
    
    Args:
        data_yaml: Path to data.yaml configuration
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Batch size for training
        pretrained: Use pretrained weights
        project: Project directory for saving results
        name: Run name
        device: Device to use ('' for auto, 'cpu', '0', '0,1' for GPUs)
        workers: Number of data loading workers
        patience: Early stopping patience
        resume: Resume training from last checkpoint
        amp: Use automatic mixed precision
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics package not installed. "
            "Run: pip install ultralytics"
        )
    
    # Generate run name
    if name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f'hull_defect_{model_size}_{timestamp}'
    
    print("=" * 60)
    print("YOLO Training for Hull Defect Detection")
    print("=" * 60)
    print(f"Model: YOLOv8{model_size}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {imgsz}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device if device else 'auto'}")
    print("=" * 60)
    
    # Initialize model
    if pretrained:
        model = YOLO(f'yolov8{model_size}.pt')
        print(f"\nLoaded pretrained YOLOv8{model_size} weights")
    else:
        model = YOLO(f'yolov8{model_size}.yaml')
        print(f"\nTraining from scratch with YOLOv8{model_size} architecture")
    
    # Training configuration
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'project': project,
        'name': name,
        'workers': workers,
        'patience': patience,
        'amp': amp,
        'verbose': True,
        'plots': True,
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        
        # Augmentation settings optimized for hull images
        'hsv_h': 0.015,      # HSV-Hue augmentation
        'hsv_s': 0.7,        # HSV-Saturation augmentation
        'hsv_v': 0.4,        # HSV-Value augmentation
        'degrees': 0.0,      # No rotation (hull images are usually aligned)
        'translate': 0.1,    # Translation
        'scale': 0.5,        # Scale augmentation
        'shear': 0.0,        # No shear
        'perspective': 0.0,  # No perspective
        'flipud': 0.0,       # No vertical flip
        'fliplr': 0.5,       # Horizontal flip
        'mosaic': 1.0,       # Mosaic augmentation
        'mixup': 0.1,        # Mixup augmentation
        'copy_paste': 0.1,   # Copy-paste augmentation
    }
    
    # Add device if specified
    if device:
        train_args['device'] = device
    
    # Resume from checkpoint
    if resume:
        train_args['resume'] = True
    
    # Start training
    print("\nStarting training...")
    results = model.train(**train_args)
    
    # Print final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best weights saved to: {project}/{name}/weights/best.pt")
    print(f"Last weights saved to: {project}/{name}/weights/last.pt")
    print(f"Training results saved to: {project}/{name}/")
    
    return results


def validate_model(
    model_path: str,
    data_yaml: str,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = ''
):
    """
    Validate a trained model on the test set.
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to data.yaml configuration
        imgsz: Input image size
        batch_size: Batch size
        device: Device to use
    """
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("Model Validation")
    print("=" * 60)
    
    model = YOLO(model_path)
    
    val_args = {
        'data': data_yaml,
        'imgsz': imgsz,
        'batch': batch_size,
        'split': 'test',
        'verbose': True,
        'plots': True,
    }
    
    if device:
        val_args['device'] = device
    
    results = model.val(**val_args)
    
    # Print metrics
    print("\nValidation Metrics:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")
    
    return results


def export_model(
    model_path: str,
    format: str = 'onnx',
    imgsz: int = 640,
    half: bool = False,
    dynamic: bool = False
):
    """
    Export trained model to various formats.
    
    Args:
        model_path: Path to trained model weights
        format: Export format (onnx, torchscript, tflite, etc.)
        imgsz: Input image size
        half: Use FP16 half precision
        dynamic: Enable dynamic axes for ONNX
    """
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print(f"Exporting Model to {format.upper()}")
    print("=" * 60)
    
    model = YOLO(model_path)
    
    export_path = model.export(
        format=format,
        imgsz=imgsz,
        half=half,
        dynamic=dynamic
    )
    
    print(f"\nModel exported to: {export_path}")
    
    return export_path


def main():
    parser = argparse.ArgumentParser(description='Train YOLO for hull defect detection')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    train_parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                             help='Model size')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    train_parser.add_argument('--batch', type=int, default=16, help='Batch size')
    train_parser.add_argument('--device', type=str, default='', help='Device to use')
    train_parser.add_argument('--no-pretrained', action='store_true', help='Train from scratch')
    train_parser.add_argument('--resume', action='store_true', help='Resume training')
    train_parser.add_argument('--name', type=str, default=None, help='Run name')
    
    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate the model')
    val_parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    val_parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    val_parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    val_parser.add_argument('--batch', type=int, default=16, help='Batch size')
    val_parser.add_argument('--device', type=str, default='', help='Device to use')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export the model')
    export_parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    export_parser.add_argument('--format', type=str, default='onnx', 
                               choices=['onnx', 'torchscript', 'tflite', 'coreml', 'openvino'],
                               help='Export format')
    export_parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    export_parser.add_argument('--half', action='store_true', help='Use FP16')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_yolo(
            data_yaml=args.data,
            model_size=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch,
            pretrained=not args.no_pretrained,
            device=args.device,
            resume=args.resume,
            name=args.name
        )
    
    elif args.command == 'validate':
        validate_model(
            model_path=args.model,
            data_yaml=args.data,
            imgsz=args.imgsz,
            batch_size=args.batch,
            device=args.device
        )
    
    elif args.command == 'export':
        export_model(
            model_path=args.model,
            format=args.format,
            imgsz=args.imgsz,
            half=args.half
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
