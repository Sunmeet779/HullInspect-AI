"""
Eigen Tree Corrosion Detection Service
=======================================
Implements the unsupervised pruned decision-tree method from:

  Chliveros, G.; Kontomaris, S.V.; Letsios, A.
  "Automatic Identification of Corrosion in Marine Vessels Using
   Decision-Tree Imaging Hierarchies"
  Eng 2023, 4, 2090-2099.  https://doi.org/10.3390/eng4030118

Algorithm summary
-----------------
1.  Treat the entire image as the root node (one cluster of all pixels).
2.  For each cluster compute the colour-channel covariance, eigenvalues,
    and eigenvectors.
3.  Pick the leaf whose eigenvalue is largest; split it into two children
    using the eigenvector as a separating hyperplane through the
    quantisation level (per-channel mean).
4.  Repeat until a target tree depth / node count is reached.
5.  Bottom-up entropy-based pruning removes redundant leaves.
6.  The dominant leaf (max eigenvalue ∩ max entropy) is predicted as the
    corrosion region.

The method is fully unsupervised — no training data required.
"""

from __future__ import annotations
import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from django.conf import settings


# ── Data structures ───────────────────────────────────────────────────────

@dataclass
class TreeNode:
    """One node in the binary eigen tree."""
    node_id: int
    pixel_indices: np.ndarray          # 1-D array of flat pixel indices
    mean_colour: np.ndarray = None     # quantisation level Q_n (3,)
    covariance: np.ndarray = None      # R̂_n (3×3)
    eigenvalue: float = 0.0            # dominant eigenvalue λ_n
    eigenvector: np.ndarray = None     # corresponding eigenvector e_n
    entropy: float = 0.0
    left: Optional[TreeNode] = None    # C_{2n}
    right: Optional[TreeNode] = None   # C_{2n+1}
    is_leaf: bool = True


# ── Core algorithm ────────────────────────────────────────────────────────

class EigenTreeCorrosionDetector:
    """
    Pruned eigen-tree hierarchy for unsupervised corrosion segmentation.

    Parameters
    ----------
    max_depth : int
        Maximum number of binary splits (paper uses 5-10).
    prune : bool
        Whether to apply bottom-up entropy pruning.
    min_cluster_pixels : int
        Don't split a cluster smaller than this.
    """

    # Severity thresholds (area %)
    SEVERITY_BY_AREA = {
        'critical': 5.0,
        'severe':   2.0,
        'moderate': 0.5,
        'mild':     0.0,
    }

    def __init__(
        self,
        max_depth: int = 7,
        prune: bool = True,
        min_cluster_pixels: int = 200,
    ):
        self.max_depth = max_depth
        self.prune = prune
        self.min_cluster_pixels = min_cluster_pixels
        self.min_leaves_after_prune = 3   # never prune below this many leaves

    # ── Public API ────────────────────────────────────────────────────

    def detect(
        self,
        image_path: str,
        save_annotated: bool = True,
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        Run eigen-tree corrosion detection on *image_path*.

        Returns
        -------
        detections : list[dict]
            Each dict mirrors the YOLO output schema so the rest of the
            Django pipeline (VLM report, results page) works unchanged.
        annotated_path : str | None
            Path to the saved annotated image, or None.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        img_h, img_w = image.shape[:2]
        total_pixels = img_h * img_w

        print(f"\n🌳 Running Eigen-Tree corrosion detection...")
        print(f"   Image: {os.path.basename(image_path)} ({img_w}×{img_h})")
        print(f"   Target leaf nodes: {self.max_depth}")

        # Flatten image to (N, 3) float
        pixels = image.reshape(-1, 3).astype(np.float64)

        # Build the full binary eigen tree
        root = self._build_tree(pixels)

        # Prune
        if self.prune:
            self._entropy_prune(root, pixels)

        # Collect surviving leaves
        leaves = self._collect_leaves(root)
        print(f"   Leaves after pruning: {len(leaves)}")

        # Select the dominant (corrosion) leaf
        dominant = self._select_dominant_leaf(leaves)
        if dominant is None:
            print("   No dominant corrosion leaf found.")
            return [], None

        # Build binary mask from dominant leaf indices
        mask = np.zeros(total_pixels, dtype=np.uint8)
        mask[dominant.pixel_indices] = 255
        mask_2d = mask.reshape(img_h, img_w)

        # Optional: morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_2d = cv2.morphologyEx(mask_2d, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_2d = cv2.morphologyEx(mask_2d, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find connected components → individual detection regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_2d, connectivity=8
        )

        detections: List[Dict] = []
        region_masks: List[np.ndarray] = []

        for lbl in range(1, num_labels):     # skip background 0
            area_px = int(stats[lbl, cv2.CC_STAT_AREA])
            if area_px < self.min_cluster_pixels:
                continue

            x1 = int(stats[lbl, cv2.CC_STAT_LEFT])
            y1 = int(stats[lbl, cv2.CC_STAT_TOP])
            w  = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h  = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            x2, y2 = x1 + w, y1 + h

            area_pct = float(area_px / total_pixels * 100)
            severity = self._severity_from_area(area_pct)

            cx = float(centroids[lbl][0])
            cy = float(centroids[lbl][1])

            region_mask = (labels == lbl).astype(np.uint8)
            region_masks.append(region_mask)

            detections.append({
                'class_id': 0,
                'class_name': 'corrosion',
                'confidence': float(dominant.eigenvalue / (dominant.eigenvalue + 1)),  # normalised proxy
                'x_center': float(cx / img_w),
                'y_center': float(cy / img_h),
                'width': float(w / img_w),
                'height': float(h / img_h),
                'bbox_pixels': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                'severity': severity,
                'area_percent': area_pct,
                'mask_pixels': area_px,
            })

        # Sort by area descending
        order = sorted(range(len(detections)),
                       key=lambda i: detections[i]['area_percent'], reverse=True)
        detections = [detections[i] for i in order]
        region_masks = [region_masks[i] for i in order]

        # Cap at 12 regions
        detections = detections[:12]
        region_masks = region_masks[:12]

        # Union area
        if region_masks:
            union = np.zeros_like(region_masks[0])
            for rm in region_masks:
                union = np.bitwise_or(union, rm)
            total_area = float(np.sum(union) / total_pixels * 100)
        else:
            total_area = 0.0

        self._last_union_area = total_area

        print(f"   Detected {len(detections)} corrosion region(s)")
        print(f"   Total corroded area: {total_area:.2f}%")

        annotated_path = None
        if save_annotated and detections:
            annotated_path = self._create_annotated_image(
                image, detections, region_masks, image_path
            )

        return detections, annotated_path

    # ── Tree construction ─────────────────────────────────────────────

    def _compute_node_stats(self, node: TreeNode, pixels: np.ndarray):
        """Compute mean, covariance, eigenvalue/vector, entropy for a node."""
        subset = pixels[node.pixel_indices]  # (k, 3)
        n = len(subset)
        if n < 2:
            return

        node.mean_colour = subset.mean(axis=0)                    # Q_n
        centered = subset - node.mean_colour
        cov = (centered.T @ centered) / n                          # R̂_n
        node.covariance = cov

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx_max = np.argmax(np.abs(eigenvalues))
        node.eigenvalue = float(np.abs(eigenvalues[idx_max]))
        node.eigenvector = eigenvectors[:, idx_max]

        # Entropy via per-channel histogram
        node.entropy = self._compute_entropy(subset)

    @staticmethod
    def _compute_entropy(subset: np.ndarray) -> float:
        """Shannon entropy over the 3-channel histogram of a pixel cluster."""
        total_h = 0.0
        for ch in range(3):
            hist, _ = np.histogram(subset[:, ch], bins=64, range=(0, 256))
            hist = hist[hist > 0].astype(np.float64)
            prob = hist / hist.sum()
            total_h -= float(np.sum(prob * np.log2(prob)))
        return total_h

    def _build_tree(self, pixels: np.ndarray) -> TreeNode:
        """
        Construct the binary eigen tree by iterative leaf splitting.

        The paper uses k = 5, 7, 10 leaf nodes.  We split until we reach
        *max_depth* leaves (not binary-tree depth).  Each iteration picks
        the leaf with the largest eigenvalue and splits it into two,
        growing the leaf count by one.
        """
        root = TreeNode(node_id=1, pixel_indices=np.arange(len(pixels)))
        self._compute_node_stats(root, pixels)

        target_leaves = self.max_depth   # e.g. 7 leaves
        for _ in range(target_leaves * 2):   # safety cap
            leaves = self._collect_leaves(root)
            if len(leaves) >= target_leaves:
                break
            # Pick the leaf with the largest eigenvalue
            best = max(leaves, key=lambda n: n.eigenvalue)
            if best.eigenvalue < 1e-6:
                break
            if len(best.pixel_indices) < 2 * self.min_cluster_pixels:
                # Try next-best leaf
                candidates = sorted(leaves, key=lambda n: n.eigenvalue, reverse=True)
                split_done = False
                for c in candidates:
                    if c.eigenvalue >= 1e-6 and len(c.pixel_indices) >= 2 * self.min_cluster_pixels:
                        self._split_node(c, pixels)
                        if not c.is_leaf:
                            split_done = True
                            break
                if not split_done:
                    break
            else:
                self._split_node(best, pixels)

        return root

    def _split_node(self, node: TreeNode, pixels: np.ndarray):
        """Binary split of a leaf node using its eigenvector."""
        subset = pixels[node.pixel_indices]
        projections = subset @ node.eigenvector
        threshold = node.mean_colour @ node.eigenvector

        mask_left = projections >= threshold
        mask_right = ~mask_left

        if mask_left.sum() < self.min_cluster_pixels or mask_right.sum() < self.min_cluster_pixels:
            return  # refuse to create tiny children

        left = TreeNode(
            node_id=node.node_id * 2,
            pixel_indices=node.pixel_indices[mask_left],
        )
        right = TreeNode(
            node_id=node.node_id * 2 + 1,
            pixel_indices=node.pixel_indices[mask_right],
        )
        self._compute_node_stats(left, pixels)
        self._compute_node_stats(right, pixels)

        node.left = left
        node.right = right
        node.is_leaf = False

    # ── Pruning ───────────────────────────────────────────────────────

    def _entropy_prune(self, root: TreeNode, pixels: np.ndarray):
        """
        Bottom-up entropy pruning (Section 2.2 of the paper).

        The paper suppresses children when both have entropy ≤ parent.
        In practice children (being subsets) almost always satisfy this,
        so we add two safety guards:
        1.  Minimum leaf count — never prune below ``min_leaves_after_prune``.
        2.  Relative entropy threshold — only prune when BOTH children
            have entropy < ``prune_ratio`` × parent entropy (default 0.7),
            meaning the split contributed very little new information.
        """
        prune_ratio = 0.70   # children must be < 70 % of parent entropy

        # Iterative bottom-up passes (at most 3 passes)
        for _ in range(3):
            leaves_before = len(self._collect_leaves(root))
            self._entropy_prune_pass(root, prune_ratio)
            leaves_after = len(self._collect_leaves(root))
            if leaves_after <= self.min_leaves_after_prune:
                break
            if leaves_after == leaves_before:
                break   # nothing changed

    def _entropy_prune_pass(self, node: TreeNode, ratio: float):
        """Single bottom-up pass."""
        if node.is_leaf:
            return

        # Recurse first (bottom-up)
        if node.left and not node.left.is_leaf:
            self._entropy_prune_pass(node.left, ratio)
        if node.right and not node.right.is_leaf:
            self._entropy_prune_pass(node.right, ratio)

        if node.left is None or node.right is None:
            return

        # Guard: count current leaves in the whole sub-tree rooted here
        # (we can't easily count global leaves, but skipping when children
        # are both leaves and it would remove 2 → net −1 leaf globally)
        # Instead we rely on the outer loop checking global count.

        parent_h = node.entropy if node.entropy > 0 else 1e-9
        if (node.left.entropy < ratio * parent_h and
                node.right.entropy < ratio * parent_h and
                node.left.is_leaf and node.right.is_leaf):
            # Suppress children — merge back into parent
            node.left = None
            node.right = None
            node.is_leaf = True

    # ── Leaf utilities ────────────────────────────────────────────────

    @staticmethod
    def _collect_leaves(node: TreeNode) -> List[TreeNode]:
        """Return all leaf nodes in the tree."""
        if node.is_leaf:
            return [node]
        leaves = []
        if node.left:
            leaves.extend(EigenTreeCorrosionDetector._collect_leaves(node.left))
        if node.right:
            leaves.extend(EigenTreeCorrosionDetector._collect_leaves(node.right))
        return leaves

    def _select_dominant_leaf(self, leaves: List[TreeNode]) -> Optional[TreeNode]:
        """
        Predict the corrosion leaf: the one with both max eigenvalue
        AND max entropy (Section 2.2).  If they disagree, pick the
        max-eigenvalue leaf (max-entropy is eliminated from candidates).

        If there is only 1 leaf (entire image), return None — we cannot
        segment anything.
        """
        if not leaves or len(leaves) < 2:
            return None

        max_eig_leaf = max(leaves, key=lambda n: n.eigenvalue)
        max_ent_leaf = max(leaves, key=lambda n: n.entropy)

        if max_eig_leaf is max_ent_leaf:
            return max_eig_leaf

        # They disagree — eliminate the max-entropy leaf and pick
        # from the remaining the one with highest eigenvalue.
        remaining = [l for l in leaves if l is not max_ent_leaf]
        if remaining:
            return max(remaining, key=lambda n: n.eigenvalue)
        return max_eig_leaf

    # ── Helpers ───────────────────────────────────────────────────────

    def _severity_from_area(self, area_pct: float) -> str:
        for sev, threshold in self.SEVERITY_BY_AREA.items():
            if area_pct >= threshold:
                return sev
        return 'mild'

    def _get_overall_severity(self, detections: List[Dict]) -> str:
        order = ['critical', 'severe', 'moderate', 'mild', 'good']
        worst = 'good'
        for d in detections:
            s = d.get('severity', 'mild')
            if s in order and order.index(s) < order.index(worst):
                worst = s
        return worst

    # ── Annotation (mirrors YOLO service output style) ────────────────

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

    MIN_ANNOTATE_DIM = 800

    def _create_annotated_image(
        self,
        image: np.ndarray,
        detections: List[Dict],
        region_masks: List[np.ndarray],
        original_path: str,
    ) -> str:
        """Create annotated image with segmentation overlays."""
        img_h, img_w = image.shape[:2]
        n = len(detections)

        # Upscale tiny images
        scale = 1.0
        short_side = min(img_h, img_w)
        if short_side < self.MIN_ANNOTATE_DIM:
            scale = self.MIN_ANNOTATE_DIM / short_side
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            new_masks = []
            for m in region_masks:
                new_masks.append(
                    cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                )
            region_masks = new_masks
            for det in detections:
                bb = det['bbox_pixels']
                bb['x1'] = int(bb['x1'] * scale)
                bb['y1'] = int(bb['y1'] * scale)
                bb['x2'] = int(bb['x2'] * scale)
                bb['y2'] = int(bb['y2'] * scale)
            img_h, img_w = new_h, new_w

        annotated = image.copy()
        ref = max(img_h, img_w)
        contour_thick = max(1, int(ref / 500))
        bbox_thick    = max(1, int(ref / 700))
        fs            = max(0.30, ref / 2400)
        font_thick    = max(1, int(ref / 1200))
        pad           = max(2, int(ref / 400))
        font          = cv2.FONT_HERSHEY_SIMPLEX

        # Mask overlay (largest first)
        draw_order = sorted(range(n), key=lambda i: detections[i]['area_percent'],
                            reverse=True)
        for i in draw_order:
            mask_bin = region_masks[i]
            color = self.DETECTION_COLORS[i % len(self.DETECTION_COLORS)]
            layer = np.full_like(annotated, color, dtype=np.uint8)
            mask_bool = mask_bin > 0
            annotated[mask_bool] = cv2.addWeighted(
                layer, 0.35, annotated, 0.65, 0
            )[mask_bool]

        # Contours
        for i in reversed(draw_order):
            mask_bin = region_masks[i]
            color = self.DETECTION_COLORS[i % len(self.DETECTION_COLORS)]
            contours, _ = cv2.findContours(
                mask_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(annotated, contours, -1, color, contour_thick)
            cv2.drawContours(annotated, contours, -1, (255, 255, 255),
                             max(1, contour_thick // 2))

        # Summary bar first (so labels avoid it)
        total_area = getattr(self, '_last_union_area',
                             min(sum(d['area_percent'] for d in detections), 100.0))
        overall_sev = self._get_overall_severity(detections)
        summary = (f"Eigen Tree | Corrosion: {n} regions | "
                   f"Total area: {total_area:.1f}% | {overall_sev.upper()}")
        sum_fs = max(0.35, ref / 2000)
        (sw, sh), _ = cv2.getTextSize(summary, font, sum_fs, 2)
        bar_h = sh + int(pad * 3)
        bar_rect = (0, 0, sw + 2 * pad + 10, bar_h)
        cv2.rectangle(annotated, (bar_rect[0], bar_rect[1]),
                      (bar_rect[2], bar_rect[3]), (0, 0, 0), -1)
        cv2.putText(annotated, summary, (pad + 4, sh + pad),
                    font, sum_fs, (0, 255, 0), max(1, font_thick), cv2.LINE_AA)

        # Labels
        used_rects: List[Tuple[int, int, int, int]] = [bar_rect]
        for i in range(n):
            det = detections[i]
            color = self.DETECTION_COLORS[i % len(self.DETECTION_COLORS)]
            bb = det['bbox_pixels']
            area = det['area_percent']
            sev  = det['severity'][0].upper()

            cv2.rectangle(annotated, (bb['x1'], bb['y1']),
                          (bb['x2'], bb['y2']), color, bbox_thick)

            label = f"#{i+1} {area:.1f}% [{sev}]"
            (tw, tht), _ = cv2.getTextSize(label, font, fs, font_thick)
            lx = max(bb['x1'], 2)
            ly = bb['y1'] - pad
            if ly - tht - pad < 0:
                ly = bb['y1'] + tht + pad + 4

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

        # Legend
        leg_fs = max(0.28, ref / 3200)
        (_, leg_h), _ = cv2.getTextSize("#", font, leg_fs, 1)
        leg_step = leg_h + pad + 2
        leg_y = img_h - pad - n * leg_step
        if leg_y < bar_h + 10:
            leg_y = bar_h + 10
        for i in range(n):
            color = self.DETECTION_COLORS[i % len(self.DETECTION_COLORS)]
            txt = f"#{i+1} {detections[i]['severity']} {detections[i]['area_percent']:.1f}%"
            sq = max(6, int(ref / 200))
            cv2.rectangle(annotated, (pad, leg_y), (pad + sq, leg_y + sq), color, -1)
            cv2.putText(annotated, txt, (pad + sq + 4, leg_y + sq - 1),
                        font, leg_fs, (255, 255, 255), 1, cv2.LINE_AA)
            leg_y += leg_step

        # Save
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        output_dir = os.path.join(settings.MEDIA_ROOT, 'uploads', 'annotated')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}_eigentree_annotated.jpg")
        cv2.imwrite(output_path, annotated)
        print(f"   Annotated image saved: {output_path}")
        return output_path
