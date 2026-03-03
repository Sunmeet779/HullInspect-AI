#!/usr/bin/env python3
"""
Evaluation Script — Paper Metrics
==================================
Computes the same metrics used in:

  Chliveros et al., "Automatic Identification of Corrosion in Marine Vessels
  Using Decision-Tree Imaging Hierarchies", Eng 2023, 4, 2090-2099.

Metrics
-------
  Recognition Accuracy : % of images where the method detected some corrosion
                         (≥ 10% of TP pixels relative to ideal coverage).
  TAC (True Area Coverage)  : TP pixels / total pixels   (mean over dataset)
  FAC (False Area Coverage) : FP pixels / total pixels   (mean over dataset)
  Performance               : TAC / (TAC + FAC)

Both YOLO11-Seg and Eigen Tree are evaluated against the MaVeCoDD ground truth.

Usage:
    python evaluate_paper_metrics.py                    # LoRes only (fast)
    python evaluate_paper_metrics.py --hires            # HiRes only
    python evaluate_paper_metrics.py --all              # Both resolutions
"""

import os
import sys
import argparse
import time
import warnings
import numpy as np
import cv2

# ── Django bootstrap ──────────────────────────────────────────────────────
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "hull_inspection.settings")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import django
django.setup()

from django.conf import settings

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
MAVECOD_DIR = os.path.join(BASE_DIR, "inspection", "ry392rp8cj-1 (1)")
LORES_RAW   = os.path.join(MAVECOD_DIR, "LoRes", "raw")
LORES_LABEL = os.path.join(MAVECOD_DIR, "LoRes", "labeled")
HIRES_RAW   = os.path.join(MAVECOD_DIR, "HiRes", "raw")
HIRES_LABEL = os.path.join(MAVECOD_DIR, "HiRes", "labeled")

# Ground-truth pixel threshold (near-black pixels are background)
GT_THRESHOLD = 10

# Recognition threshold: method must detect ≥ this fraction of GT corrosion
# pixels to count as "recognised" for that image.
RECOGNITION_THRESHOLD = 0.10


# ── Helpers ───────────────────────────────────────────────────────────────

def load_ground_truth(label_path: str, shape_hw: tuple) -> np.ndarray:
    """
    Load a MaVeCoDD ground-truth label image and return a binary mask.
    Non-zero (> GT_THRESHOLD in any channel) pixels are corrosion.
    """
    label = cv2.imread(label_path)
    if label is None:
        raise FileNotFoundError(f"Cannot read label: {label_path}")
    # Resize if dimensions don't match raw image
    if label.shape[:2] != shape_hw:
        label = cv2.resize(label, (shape_hw[1], shape_hw[0]),
                           interpolation=cv2.INTER_NEAREST)
    mask = np.any(label > GT_THRESHOLD, axis=2).astype(np.uint8)
    return mask


def find_label_file(label_dir: str, stem: str) -> str | None:
    """Find the matching label file accounting for naming variations."""
    candidates = [
        f"{stem}_label.jpg",
        f"{stem}_labels.jpg",
        f"{stem}_label.png",
        f"{stem}_labels.png",
    ]
    for c in candidates:
        p = os.path.join(label_dir, c)
        if os.path.exists(p):
            return p
    return None


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    Compute paper metrics for a single image.

    Returns dict with tp_pixels, fp_pixels, total_pixels,
    tac, fac, ideal_coverage, and recognised (bool).
    """
    total = gt_mask.shape[0] * gt_mask.shape[1]
    gt_bool = gt_mask.astype(bool)
    pred_bool = pred_mask.astype(bool)

    tp = int(np.count_nonzero(pred_bool & gt_bool))
    fp = int(np.count_nonzero(pred_bool & ~gt_bool))

    ideal = int(np.count_nonzero(gt_bool))
    ideal_coverage = ideal / total * 100

    tac = tp / total * 100
    fac = fp / total * 100

    # Recognition: TP ≥ RECOGNITION_THRESHOLD of ideal coverage pixels
    recognised = tp >= (RECOGNITION_THRESHOLD * ideal) if ideal > 0 else False

    return {
        "tp": tp,
        "fp": fp,
        "total": total,
        "tac": tac,
        "fac": fac,
        "ideal_coverage": ideal_coverage,
        "recognised": recognised,
    }


# ── YOLO prediction ──────────────────────────────────────────────────────

def predict_yolo(image_path: str, yolo_service) -> np.ndarray:
    """
    Run YOLO inference and return a binary union-mask (uint8, 0/1).
    """
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    if yolo_service.model is None:
        yolo_service.load_model()

    results = yolo_service.model(
        image,
        conf=yolo_service.conf_threshold,
        iou=yolo_service.iou_threshold,
        verbose=False,
        retina_masks=True,
    )
    # Note: model loading messages come from the YOLO service;
    # they are suppressed via stdout redirect in evaluate_dataset().

    union = np.zeros((h, w), dtype=np.uint8)

    for result in results:
        if result.masks is None:
            continue
        for i in range(len(result.boxes)):
            mask = result.masks.data[i].cpu().numpy()
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            union = np.bitwise_or(union, (mask > 0.5).astype(np.uint8))

    return union


# ── Eigen Tree prediction ────────────────────────────────────────────────

def predict_eigen_tree(image_path: str, eigen_service) -> np.ndarray:
    """
    Run Eigen Tree detection and return a binary union-mask (uint8, 0/1).
    """
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    total_pixels = h * w

    pixels = image.reshape(-1, 3).astype(np.float64)

    # Build tree
    root = eigen_service._build_tree(pixels)

    # Prune
    if eigen_service.prune:
        eigen_service._entropy_prune(root, pixels)

    leaves = eigen_service._collect_leaves(root)
    dominant = eigen_service._select_dominant_leaf(leaves)

    if dominant is None:
        return np.zeros((h, w), dtype=np.uint8)

    mask_flat = np.zeros(total_pixels, dtype=np.uint8)
    mask_flat[dominant.pixel_indices] = 1
    mask_2d = mask_flat.reshape(h, w)

    # Morphological cleanup (same as the service does)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_2d = cv2.morphologyEx(mask_2d, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_2d = cv2.morphologyEx(mask_2d, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask_2d


# ── Main evaluation loop ─────────────────────────────────────────────────

def evaluate_dataset(raw_dir: str, label_dir: str, tag: str):
    """
    Evaluate both methods on a single resolution set and print results.
    """
    import io
    import contextlib
    from inspection.services.yolo_seg_inference import YOLOSegInferenceService
    from inspection.services.eigen_tree_service import EigenTreeCorrosionDetector

    # Suppress internal print() from services (model loading, etc.)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yolo_svc = YOLOSegInferenceService()
        eigen_svc = EigenTreeCorrosionDetector(max_depth=7, prune=True)
        # Pre-load model so loading messages don't leak into output
        if yolo_svc.model is None:
            yolo_svc.load_model()

    files = sorted(
        f for f in os.listdir(raw_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not files:
        print(f"  No images found in {raw_dir}")
        return

    yolo_results = []
    eigen_results = []

    print(f"\n{'='*72}")
    print(f" Evaluating {tag} — {len(files)} images")
    print(f"{'='*72}")
    print(f"{'Image':<30} {'Ideal%':>7} │ {'YOLO TAC':>8} {'FAC':>7} {'Perf':>6} │ {'ET TAC':>8} {'FAC':>7} {'Perf':>6}")
    print(f"{'-'*30} {'-'*7} ┼ {'-'*8} {'-'*7} {'-'*6} ┼ {'-'*8} {'-'*7} {'-'*6}")

    for fname in files:
        raw_path = os.path.join(raw_dir, fname)
        stem = os.path.splitext(fname)[0]
        label_path = find_label_file(label_dir, stem)

        if label_path is None:
            print(f"  {fname:<30} ⚠ no label found — skipping")
            continue

        image = cv2.imread(raw_path)
        if image is None:
            continue
        h, w = image.shape[:2]

        gt = load_ground_truth(label_path, (h, w))

        # ── YOLO ──
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                yolo_mask = predict_yolo(raw_path, yolo_svc)
            ym = compute_metrics(yolo_mask, gt)
            yolo_results.append(ym)
        except Exception as e:
            print(f"  {fname:<30} YOLO error: {e}")
            ym = None

        # ── Eigen Tree ──
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                et_mask = predict_eigen_tree(raw_path, eigen_svc)
            em = compute_metrics(et_mask, gt)
            eigen_results.append(em)
        except Exception as e:
            print(f"  {fname:<30} ET error: {e}")
            em = None

        # ── Print row ──
        ideal = ym["ideal_coverage"] if ym else (em["ideal_coverage"] if em else 0)

        def perf(m):
            if m is None:
                return "   err"
            p = m["tac"] / (m["tac"] + m["fac"]) * 100 if (m["tac"] + m["fac"]) > 0 else 0
            return f"{p:5.1f}%"

        def fmt(m, key):
            return f"{m[key]:6.2f}%" if m else "   err"

        print(
            f"  {fname:<28} {ideal:6.2f}% │ "
            f"{fmt(ym, 'tac')} {fmt(ym, 'fac')} {perf(ym)} │ "
            f"{fmt(em, 'tac')} {fmt(em, 'fac')} {perf(em)}"
        )

    # ── Summary ───────────────────────────────────────────────────────
    def summarise(name, results):
        if not results:
            return
        n = len(results)
        rec_acc = sum(1 for r in results if r["recognised"]) / n * 100
        mean_tac = np.mean([r["tac"] for r in results])
        mean_fac = np.mean([r["fac"] for r in results])
        perf = mean_tac / (mean_tac + mean_fac) * 100 if (mean_tac + mean_fac) > 0 else 0
        mean_ideal = np.mean([r["ideal_coverage"] for r in results])

        print(f"\n  {name} — {n} images evaluated")
        print(f"    Recognition Accuracy : {rec_acc:.1f}%")
        print(f"    Mean Ideal Coverage  : {mean_ideal:.1f}%")
        print(f"    Mean TAC (TP area)   : {mean_tac:.1f}%")
        print(f"    Mean FAC (FP area)   : {mean_fac:.1f}%")
        print(f"    Performance          : {perf:.1f}%")

    print(f"\n{'─'*72}")
    print(f" SUMMARY — {tag}")
    print(f"{'─'*72}")
    summarise("YOLO11-Seg", yolo_results)
    summarise("Eigen Tree (k=7, pruned)", eigen_results)

    # ── Paper comparison ──
    print(f"\n{'─'*72}")
    print(f" COMPARISON WITH PAPER (Chliveros et al. 2023)")
    print(f"{'─'*72}")
    print(f"  {'Method':<30} {'Rec. Acc.':>10} {'TAC (TP%)':>10} {'FAC (FP%)':>10} {'Perf':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    print(f"  {'CNN U-Net (paper)':<30} {'96%':>10} {'15%':>10} {'20%':>10} {'44%':>8}")
    print(f"  {'SOM 4x2 (paper)':<30} {'95%':>10} {'9%':>10} {'10%':>10} {'47%':>8}")
    print(f"  {'Pruned ET k=7 (paper)':<30} {'94%':>10} {'13.4%':>10} {'5.9%':>10} {'56%':>8}")

    if yolo_results:
        n = len(yolo_results)
        rec = sum(1 for r in yolo_results if r["recognised"]) / n * 100
        tac = np.mean([r["tac"] for r in yolo_results])
        fac = np.mean([r["fac"] for r in yolo_results])
        p = tac / (tac + fac) * 100 if (tac + fac) > 0 else 0
        print(f"  {'YOLO11-Seg (ours)':<30} {rec:>9.0f}% {tac:>9.1f}% {fac:>9.1f}% {p:>7.0f}%")

    if eigen_results:
        n = len(eigen_results)
        rec = sum(1 for r in eigen_results if r["recognised"]) / n * 100
        tac = np.mean([r["tac"] for r in eigen_results])
        fac = np.mean([r["fac"] for r in eigen_results])
        p = tac / (tac + fac) * 100 if (tac + fac) > 0 else 0
        print(f"  {'Eigen Tree k=7 (ours)':<30} {rec:>9.0f}% {tac:>9.1f}% {fac:>9.1f}% {p:>7.0f}%")

    print()


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO & Eigen Tree with paper metrics")
    parser.add_argument("--hires", action="store_true", help="Evaluate HiRes images only")
    parser.add_argument("--all", action="store_true", help="Evaluate both LoRes and HiRes")
    args = parser.parse_args()

    start = time.time()

    if args.hires:
        evaluate_dataset(HIRES_RAW, HIRES_LABEL, "HiRes (3799×2256)")
    elif args.all:
        evaluate_dataset(LORES_RAW, LORES_LABEL, "LoRes (1920×1080)")
        evaluate_dataset(HIRES_RAW, HIRES_LABEL, "HiRes (3799×2256)")
    else:
        evaluate_dataset(LORES_RAW, LORES_LABEL, "LoRes (1920×1080)")

    elapsed = time.time() - start
    print(f"Total evaluation time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
