"""
ferif.py

FERIF-S: Scale-Invariant Fast Estimate Robust Image Features.

У цій версії:
- будується scale-space (кілька Gaussian-рівнів),
- WIF виявляються на різних sigma,
- патч дескриптора нормалізується до канонічного розміру,
- орієнтація оцінюється SIFT-подібною гістограмою,
- формується градієнтний дескриптор (4x4x8 = 128 компонент).

FERIFDetector є псевдонімом FERIF-S для сумісності.
"""

from typing import List, Tuple, Optional

import cv2
import numpy as np


class FERIFScaleInvariantDetector:
    """
    FERIF-S: scale-invariant варіант FERIF.

    Основні ідеї:
    - Multi-sigma Gaussian blur (scale-space) без зміни роздільної здатності.
    - WIF (weak information features) шукаються методом goodFeaturesToTrack
      на масці великих градієнтів для кожної sigma.
    - Для кожної точки (x, y, sigma) відбувається уточнення (refinement)
      на відповідному масштабі, оцінка орієнтації та побудова дескриптора
      на канонічному патчі (scale-normalized patch).
    """

    def __init__(
        self,
        # Масштабні рівні (Gaussian sigma, у пікселях)
        sigmas: Tuple[float, ...] = (1.0, 1.6, 2.3, 3.2),
        grad_percentile: float = 70.0,
        max_wif_per_scale: int = 400,
        min_wif_distance: float = 4.0,
        refine_radius: int = 3,
        edge_ratio_thresh: float = 12.0,
        min_grad_energy: float = 1.5,
        # Канонічний патч і дескриптор
        canonical_patch_size: int = 32,   # 32x32
        patch_radius_base: int = 8,       # масштабний радіус у базовій sigma_ref
        sigma_ref: float = 1.6,           # "базова" sigma для масштабування патча
        grid_size: int = 4,               # 4x4 клітинки
        orient_bins: int = 8,
        max_keypoints: int = 600,         # глобальний максимум ключових точок
    ):
        self.sigmas = sigmas
        self.grad_percentile = grad_percentile
        self.max_wif_per_scale = max_wif_per_scale
        self.min_wif_distance = min_wif_distance
        self.refine_radius = refine_radius
        self.edge_ratio_thresh = edge_ratio_thresh
        self.min_grad_energy = min_grad_energy

        self.canonical_patch_size = canonical_patch_size
        self.patch_radius_base = patch_radius_base
        self.sigma_ref = sigma_ref
        self.grid_size = grid_size
        self.orient_bins = orient_bins
        self.max_keypoints = max_keypoints

        # Попередньо обчислюємо Гауссову «шапку» для ваг у дескрипторі
        self.gaussian_weights = self._make_gaussian_weights(canonical_patch_size)

    # ------------------------------------------------------------------
    # Публічний інтерфейс
    # ------------------------------------------------------------------
    def detect_and_compute(
        self, image: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Детекція ключових точок та обчислення FERIF-S дескрипторів.

        Args:
            image: BGR або grayscale

        Returns:
            keypoints: list[cv2.KeyPoint]
            descriptors: np.ndarray (N, D) або None
        """
        if image is None:
            return [], None

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = gray.astype(np.float32)
        h, w = gray.shape

        # Побудова scale-space (Gaussian blur для кожної sigma)
        scales = []
        for sigma in self.sigmas:
            blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
            gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            ang = np.degrees(np.arctan2(gy, gx))
            scales.append(
                {
                    "sigma": sigma,
                    "blur": blur,
                    "gx": gx,
                    "gy": gy,
                    "mag": mag,
                    "ang": ang,
                }
            )

        # 1-й етап: виявлення WIF на кожному масштабі
        wif_candidates = self._detect_wif_multi_scale(scales, h, w)
        if not wif_candidates:
            return [], None

        # 2-й етап: уточнення та побудова дескрипторів
        keypoints: List[cv2.KeyPoint] = []
        descriptors: List[np.ndarray] = []

        margin = self._max_required_margin()

        for (x0, y0, s_idx) in wif_candidates:
            if len(keypoints) >= self.max_keypoints:
                break

            x0_i = int(round(x0))
            y0_i = int(round(y0))

            if (
                x0_i < margin
                or x0_i >= w - margin
                or y0_i < margin
                or y0_i >= h - margin
            ):
                continue

            scale_data = scales[s_idx]

            refined = self._refine_keypoint_scale(
                scale_data["gx"],
                scale_data["gy"],
                scale_data["mag"],
                scale_data["ang"],
                x0_i,
                y0_i,
            )
            if refined is None:
                continue

            x_ref, y_ref, angle_deg = refined

            # Перевірка з урахуванням уточнених координат
            xr = int(round(x_ref))
            yr = int(round(y_ref))
            if (
                xr < margin
                or xr >= w - margin
                or yr < margin
                or yr >= h - margin
            ):
                continue

            sigma = scale_data["sigma"]

            desc = self._build_descriptor_scale(
                scale_data["mag"],
                scale_data["ang"],
                xr,
                yr,
                angle_deg,
                sigma,
            )
            if desc is None:
                continue

            kp = cv2.KeyPoint(
                float(x_ref),
                float(y_ref),
                float(self.canonical_patch_size),  # size (умовний)
                float(angle_deg),
            )

            keypoints.append(kp)
            descriptors.append(desc.astype(np.float32))

        if not descriptors:
            return [], None

        desc_mat = np.vstack(descriptors).astype(np.float32)
        return keypoints, desc_mat

    # ------------------------------------------------------------------
    # 1-й етап: WIF у scale-space
    # ------------------------------------------------------------------
    def _detect_wif_multi_scale(
        self,
        scales: list,
        h: int,
        w: int,
    ) -> List[Tuple[float, float, int]]:
        """
        Виявлення weak information features (WIF) на кожному масштабі.
        Повертає список (x, y, scale_index).
        """
        candidates: List[Tuple[float, float, int]] = []

        for s_idx, sdata in enumerate(scales):
            mag = sdata["mag"]
            blur = sdata["blur"]

            thr = np.percentile(mag, self.grad_percentile)
            mask = (mag >= thr).astype(np.uint8) * 255

            corners = cv2.goodFeaturesToTrack(
                blur.astype(np.uint8),
                maxCorners=self.max_wif_per_scale,
                qualityLevel=0.01,
                minDistance=self.min_wif_distance,
                mask=mask,
            )

            if corners is not None:
                for c in corners:
                    x, y = c.ravel()
                    # координати в межах кадру
                    if 0 <= x < w and 0 <= y < h:
                        candidates.append((float(x), float(y), s_idx))

        # Просторова NMS з урахуванням масштабу (просто ігноруємо sigma при NMS)
        if not candidates:
            return []

        return self._non_max_suppress_points_with_scale(
            candidates, radius=3.0
        )

    @staticmethod
    def _non_max_suppress_points_with_scale(
        points: List[Tuple[float, float, int]],
        radius: float = 3.0,
    ) -> List[Tuple[float, float, int]]:
        """
        Просторе NMS для списку (x, y, s_idx).
        """
        if not points:
            return []

        pts = np.array([(p[0], p[1]) for p in points], dtype=np.float32)
        kept: List[Tuple[float, float, int]] = []
        used = np.zeros(len(pts), dtype=bool)

        for i in range(len(pts)):
            if used[i]:
                continue
            pi = pts[i]
            kept.append((float(pi[0]), float(pi[1]), points[i][2]))
            d2 = np.sum((pts - pi) ** 2, axis=1)
            used |= d2 <= (radius * radius)

        return kept

    # ------------------------------------------------------------------
    # 2-й етап: уточнення WIF та орієнтація на відповідному масштабі
    # ------------------------------------------------------------------
    def _refine_keypoint_scale(
        self,
        gx: np.ndarray,
        gy: np.ndarray,
        mag: np.ndarray,
        ang: np.ndarray,
        x0: int,
        y0: int,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Уточнення точки на конкретному масштабі:
        - пошук локального максимуму магнітуди,
        - структура градієнтів для edge-фільтра,
        - оцінка основної орієнтації.
        """
        r = self.refine_radius
        h, w = mag.shape

        if x0 < r or x0 >= w - r or y0 < r or y0 >= h - r:
            return None

        patch_mag = mag[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]
        patch_gx = gx[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]
        patch_gy = gy[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]
        patch_ang = ang[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]

        # локальний максимум магнітуди
        dy_idx, dx_idx = np.unravel_index(
            np.argmax(patch_mag), patch_mag.shape
        )
        x_ref = x0 + (dx_idx - r)
        y_ref = y0 + (dy_idx - r)

        energy = float(np.mean(patch_mag))
        if energy < self.min_grad_energy:
            return None

        # структурний тензор
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

        # орієнтація: 36-бінова гістограма
        ang_flat = patch_ang.flatten()
        mag_flat = patch_mag.flatten()
        hist, bin_edges = np.histogram(
            ang_flat,
            bins=36,
            range=(-180.0, 180.0),
            weights=mag_flat,
        )
        main_bin = int(np.argmax(hist))
        # проста оцінка центру біну
        angle_deg = (bin_edges[main_bin] + bin_edges[main_bin + 1]) / 2.0

        return x_ref, y_ref, angle_deg

    # ------------------------------------------------------------------
    # Побудова дескриптора зі scale-normalized patch
    # ------------------------------------------------------------------
    def _build_descriptor_scale(
        self,
        mag: np.ndarray,
        ang: np.ndarray,
        x: int,
        y: int,
        key_angle_deg: float,
        sigma: float,
    ) -> Optional[np.ndarray]:
        """
        Побудова FERIF-S дескриптора:
        - витягується патч радіусом ~ patch_radius_base * (sigma / sigma_ref),
        - патч ресемплюється до canonical_patch_size x canonical_patch_size,
        - орієнтації беруться відносно key_angle_deg,
        - використовуються Gaussian-весові для стабілізації,
        - формується гістограма (grid_size x grid_size x orient_bins).
        """
        h, w = mag.shape

        # масштабний радіус у пікселях
        radius_pix = int(
            max(4, round(self.patch_radius_base * (sigma / self.sigma_ref)))
        )
        if (
            x < radius_pix
            or x >= w - radius_pix
            or y < radius_pix
            or y >= h - radius_pix
        ):
            return None

        patch_mag = mag[y - radius_pix : y + radius_pix + 1,
                        x - radius_pix : x + radius_pix + 1]
        patch_ang = ang[y - radius_pix : y + radius_pix + 1,
                        x - radius_pix : x + radius_pix + 1]

        # Нормалізуємо патч до канонічного розміру
        P = self.canonical_patch_size
        patch_mag_n = cv2.resize(
            patch_mag, (P, P), interpolation=cv2.INTER_LINEAR
        )
        patch_ang_n = cv2.resize(
            patch_ang, (P, P), interpolation=cv2.INTER_LINEAR
        )

        # Орієнтації відносно key_angle_deg
        rel_ang = patch_ang_n - key_angle_deg
        rel_ang = (rel_ang + 180.0) % 360.0 - 180.0

        # Gaussian weighting (як у SIFT)
        patch_mag_w = patch_mag_n * self.gaussian_weights

        # Розбиття на сітку
        cell_size = P // self.grid_size
        if cell_size <= 0:
            return None

        desc_parts: List[np.ndarray] = []

        for gy_idx in range(self.grid_size):
            for gx_idx in range(self.grid_size):
                y0 = gy_idx * cell_size
                x0 = gx_idx * cell_size
                y1 = y0 + cell_size
                x1 = x0 + cell_size

                cell_mag = patch_mag_w[y0:y1, x0:x1].flatten()
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

        # L2-нормалізація + обрізання
        norm = np.linalg.norm(desc_vec)
        if norm > 1e-6:
            desc_vec /= norm
            desc_vec = np.clip(desc_vec, 0.0, 0.2)
            norm2 = np.linalg.norm(desc_vec)
            if norm2 > 1e-6:
                desc_vec /= norm2

        return desc_vec

    # ------------------------------------------------------------------
    # Допоміжні функції
    # ------------------------------------------------------------------
    @staticmethod
    def _make_gaussian_weights(size: int, sigma_factor: float = 0.5) -> np.ndarray:
        """
        Гауссові ваги для стабілізації дескриптора.
        sigma ≈ sigma_factor * size
        """
        coords = np.arange(size, dtype=np.float32)
        cx = (size - 1) / 2.0
        cy = (size - 1) / 2.0
        xx, yy = np.meshgrid(coords, coords)
        dx = xx - cx
        dy = yy - cy
        sigma = sigma_factor * size
        g = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        g /= g.max()
        return g.astype(np.float32)

    def _max_required_margin(self) -> int:
        """
        Оцінка максимального відступу від краю, який потрібен для
        витягування масштабного патча.
        """
        max_sigma = max(self.sigmas) if self.sigmas else 1.0
        radius_pix = int(
            max(4, round(self.patch_radius_base * (max_sigma / self.sigma_ref)))
        )
        return radius_pix + self.refine_radius + 2


# ----------------------------------------------------------------------
# Псевдоніми для сумісності
# ----------------------------------------------------------------------


class FERIFDetector(FERIFScaleInvariantDetector):
    """
    Для зворотної сумісності: FERIFDetector = FERIF-S.
    """
    pass


def ferif_s_detect(image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Зручна функція для виклику FERIF-S у стилі detectAndCompute.
    """
    detector = FERIFScaleInvariantDetector()
    return detector.detect_and_compute(image)


def ferif_detect(image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Сумісне ім'я: за замовчуванням також використовує FERIF-S.
    """
    detector = FERIFScaleInvariantDetector()
    return detector.detect_and_compute(image)
