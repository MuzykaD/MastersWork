"""
ferif.py

Implementation of FERIF: Fast Estimate Robust Image Features.

This module provides a FERIFDetector class and a convenience
function ferif_detect(image) compatible with OpenCV-style
detectAndCompute interfaces.
"""

from typing import List, Tuple, Optional

import cv2
import numpy as np


class FERIFDetector:
    def __init__(
        self,
        num_decimations: int = 2,
        decimation_factor: float = 0.5,
        median_ksize: int = 5,
        max_wif_per_level: int = 500,
        min_wif_distance: float = 4.0,
        grad_percentile: float = 85.0,
        refine_radius: int = 3,
        edge_ratio_thresh: float = 10.0,
        min_grad_energy: float = 5.0,
        patch_radius: int = 8,
        grid_size: int = 4,
        orient_bins: int = 8,
    ):
        """
        FERIFDetector parameters.

        num_decimations: number of multi-scale decimation levels (1–3)
        decimation_factor: scaling factor < 1.0 for each decimation
        median_ksize: kernel size for median blur in the pre-processor
        max_wif_per_level: maximum weak features (WIF) per scale level
        min_wif_distance: minimum distance between WIF points
        grad_percentile: gradient magnitude percentile for WIF mask
        refine_radius: local neighborhood radius for refinement step
        edge_ratio_thresh: eigenvalue ratio threshold for edge-like rejection
        min_grad_energy: minimum average gradient magnitude in refinement patch
        patch_radius: radius of descriptor patch (patch size = 2 * radius)
        grid_size: number of cells per side in descriptor grid
        orient_bins: number of orientation bins per cell
        """
        self.num_decimations = num_decimations
        self.decimation_factor = decimation_factor
        self.median_ksize = median_ksize
        self.max_wif_per_level = max_wif_per_level
        self.min_wif_distance = min_wif_distance
        self.grad_percentile = grad_percentile
        self.refine_radius = refine_radius
        self.edge_ratio_thresh = edge_ratio_thresh
        self.min_grad_energy = min_grad_energy
        self.patch_radius = patch_radius
        self.grid_size = grid_size
        self.orient_bins = orient_bins

    # ------------------------------------------------------------------
    # Public interface: similar to OpenCV Feature2D.detectAndCompute
    # ------------------------------------------------------------------
    def detect_and_compute(
        self, image: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Detect keypoints and compute FERIF descriptors.

        Args:
            image: input image (BGR or grayscale)

        Returns:
            keypoints: list of cv2.KeyPoint
            descriptors: np.ndarray of shape (N, D) or None
        """
        if image is None:
            return [], None

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = gray.astype(np.float32)

        # Base smoothing
        base = cv2.GaussianBlur(gray, (3, 3), 0)

        # Precompute gradients of the base image
        base_gx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
        base_gy = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)
        base_mag = cv2.magnitude(base_gx, base_gy)
        base_ang = np.degrees(np.arctan2(base_gy, base_gx))

        # 1st cycle: multi-scale decimation and WIF detection
        wif_points = self._detect_wif_multi_scale(base)
        if not wif_points:
            return [], None

        # 2nd cycle: refinement and RIF descriptor construction
        keypoints: List[cv2.KeyPoint] = []
        descriptors: List[np.ndarray] = []

        h, w = base.shape
        margin = self.patch_radius + self.refine_radius + 1

        for (x0, y0) in wif_points:
            x0 = int(round(x0))
            y0 = int(round(y0))

            if x0 < margin or x0 >= w - margin or y0 < margin or y0 >= h - margin:
                continue

            refined = self._refine_keypoint(
                base, base_gx, base_gy, base_mag, base_ang, x0, y0
            )
            if refined is None:
                continue

            x_ref, y_ref, angle_deg = refined

            xr = int(round(x_ref))
            yr = int(round(y_ref))
            if xr < margin or xr >= w - margin or yr < margin or yr >= h - margin:
                continue

            desc = self._build_descriptor(base_mag, base_ang, xr, yr, angle_deg)
            if desc is None:
                continue

            # IMPORTANT: use positional arguments only (no _size, _angle)
            kp = cv2.KeyPoint(
                float(x_ref),                  # x
                float(y_ref),                  # y
                float(self.patch_radius * 2),  # size
                float(angle_deg),              # angle
            )

            keypoints.append(kp)
            descriptors.append(desc.astype(np.float32))

        if not descriptors:
            return [], None

        desc_mat = np.vstack(descriptors).astype(np.float32)
        return keypoints, desc_mat

    # ------------------------------------------------------------------
    # 1st cycle: multi-scale decimation and WIF detection
    # ------------------------------------------------------------------
    def _detect_wif_multi_scale(self, base: np.ndarray) -> List[Tuple[float, float]]:
        """
        Multi-scale detection of weak information features (WIF) using
        gradient-based masks and goodFeaturesToTrack over decimated images.
        """
        current = base.copy()
        scale_factor = 1.0
        all_points: List[Tuple[float, float]] = []

        for level in range(self.num_decimations):
            gx = cv2.Sobel(current, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(current, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)

            thr = np.percentile(mag, self.grad_percentile)
            mask = (mag >= thr).astype(np.uint8) * 255

            corners = cv2.goodFeaturesToTrack(
                current.astype(np.uint8),
                maxCorners=self.max_wif_per_level,
                qualityLevel=0.01,
                minDistance=self.min_wif_distance,
                mask=mask,
            )

            if corners is not None:
                for c in corners:
                    x, y = c.ravel()
                    base_x = x / scale_factor
                    base_y = y / scale_factor
                    all_points.append((base_x, base_y))

            if level < self.num_decimations - 1:
                current_u8 = current.astype(np.uint8)
                blurred = cv2.medianBlur(current_u8, self.median_ksize)
                blurred = blurred.astype(np.float32)

                new_w = max(8, int(blurred.shape[1] * self.decimation_factor))
                new_h = max(8, int(blurred.shape[0] * self.decimation_factor))
                if new_w < 8 or new_h < 8:
                    break

                current = cv2.resize(
                    blurred, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
                scale_factor *= self.decimation_factor

        if not all_points:
            return []

        all_points = self._non_max_suppress_points(all_points, radius=3.0)
        return all_points

    @staticmethod
    def _non_max_suppress_points(
        points: List[Tuple[float, float]], radius: float = 3.0
    ) -> List[Tuple[float, float]]:
        """
        Simple spatial non-maximum suppression on a set of candidate points.
        """
        if not points:
            return []

        pts = np.array(points, dtype=np.float32)
        kept: List[Tuple[float, float]] = []
        used = np.zeros(len(pts), dtype=bool)

        for i in range(len(pts)):
            if used[i]:
                continue
            pi = pts[i]
            kept.append((float(pi[0]), float(pi[1])))
            d2 = np.sum((pts - pi) ** 2, axis=1)
            used |= d2 <= (radius * radius)

        return kept

    # ------------------------------------------------------------------
    # 2nd cycle: refinement and RIF detection
    # ------------------------------------------------------------------
    def _refine_keypoint(
        self,
        base: np.ndarray,
        gx: np.ndarray,
        gy: np.ndarray,
        mag: np.ndarray,
        ang: np.ndarray,
        x0: int,
        y0: int,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Refine a WIF location:
        - shift to local maximum of gradient magnitude;
        - compute gradient structure tensor to reject edge-like points;
        - estimate dominant orientation via orientation histogram;
        - check local gradient energy.
        """
        r = self.refine_radius
        h, w = base.shape

        if x0 < r or x0 >= w - r or y0 < r or y0 >= h - r:
            return None

        patch_mag = mag[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]
        patch_gx = gx[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]
        patch_gy = gy[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]
        patch_ang = ang[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]

        dy_idx, dx_idx = np.unravel_index(
            np.argmax(patch_mag), patch_mag.shape
        )
        x_ref = x0 + (dx_idx - r)
        y_ref = y0 + (dy_idx - r)

        energy = float(np.mean(patch_mag))
        if energy < self.min_grad_energy:
            return None

        gxx = float(np.sum(patch_gx * patch_gx))
        gyy = float(np.sum(patch_gy * patch_gy))
        gxy = float(np.sum(patch_gx * patch_gy))

        trace = gxx + gyy
        det = gxx * gyy - gxy * gxy
        if det <= 1e-12:
            return None

        tmp = max(trace * trace / 4.0 - det, 0.0)
        sqrt_term = float(np.sqrt(tmp))
        lambda1 = trace / 2.0 + sqrt_term
        lambda2 = trace / 2.0 - sqrt_term
        if lambda2 <= 1e-12:
            return None

        ratio = lambda1 / lambda2
        if ratio > self.edge_ratio_thresh:
            return None

        ang_flat = patch_ang.flatten()
        mag_flat = patch_mag.flatten()
        hist, bin_edges = np.histogram(
            ang_flat,
            bins=36,
            range=(-180.0, 180.0),
            weights=mag_flat,
        )
        main_bin = int(np.argmax(hist))
        angle_deg = (bin_edges[main_bin] + bin_edges[main_bin + 1]) / 2.0

        return x_ref, y_ref, angle_deg

    # ------------------------------------------------------------------
    # Descriptor construction (SIFT-like RIF)
    # ------------------------------------------------------------------
    def _build_descriptor(
        self,
        mag: np.ndarray,
        ang: np.ndarray,
        x: int,
        y: int,
        key_angle_deg: float,
    ) -> Optional[np.ndarray]:
        """
        Build a FERIF descriptor:
        - square patch of size (2 * patch_radius)^2;
        - grid_size × grid_size cells;
        - orient_bins histogram bins per cell;
        - orientations are relative to keypoint orientation.
        """
        r = self.patch_radius
        h, w = mag.shape

        if x < r or x >= w - r or y < r or y >= h - r:
            return None

        patch_mag = mag[y - r : y + r, x - r : x + r]
        patch_ang = ang[y - r : y + r, x - r : x + r]

        rel_ang = patch_ang - key_angle_deg
        rel_ang = (rel_ang + 180.0) % 360.0 - 180.0

        step = (2 * r) // self.grid_size
        if step <= 0:
            return None

        desc_parts: List[np.ndarray] = []

        for gy_idx in range(self.grid_size):
            for gx_idx in range(self.grid_size):
                y0 = gy_idx * step
                x0 = gx_idx * step
                y1 = y0 + step
                x1 = x0 + step

                cell_mag = patch_mag[y0:y1, x0:x1].flatten()
                cell_ang = rel_ang[y0:y1, x0:x1].flatten()

                if cell_mag.size == 0:
                    hist = np.zeros(self.orient_bins, dtype=np.float32)
                else:
                    hist, _ = np.histogram(
                        cell_ang,
                        bins=self.orient_bins,
                        range=(-180.0, 180.0),
                        weights=cell_mag,
                    )
                    hist = hist.astype(np.float32)

                desc_parts.append(hist)

        desc_vec = np.concatenate(desc_parts).astype(np.float32)

        norm = np.linalg.norm(desc_vec)
        if norm > 1e-6:
            desc_vec /= norm
            desc_vec = np.clip(desc_vec, 0.0, 0.2)
            norm2 = np.linalg.norm(desc_vec)
            if norm2 > 1e-6:
                desc_vec /= norm2

        return desc_vec


def ferif_detect(image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Convenience function to use FERIFDetector in a functional style.
    """
    detector = FERIFDetector()
    return detector.detect_and_compute(image)
