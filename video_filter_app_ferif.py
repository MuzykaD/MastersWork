import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import numpy as np
from typing import Optional, Callable, List, Dict, Tuple
import time
from datetime import datetime
import os
from ferif import ferif_detect, FERIFDetector

# ============================================================================
# CONFIGURATION - Adjust these settings as needed
# ============================================================================

# Feature Detection Settings
MAX_KEYPOINTS_ORB = 1000      # Maximum keypoints for ORB detector (higher = more features, slower)
MAX_KEYPOINTS_SIFT = 1500     # Maximum keypoints for SIFT detector
MAX_KEYPOINTS_SURF = 400     # Maximum keypoints for SURF detector

# Gallery Settings
MAX_SAVED_FRAMES = 50        # Maximum number of frames to keep in gallery

# Video Display Settings
MIN_WINDOW_WIDTH = 1000      # Minimum window width in pixels
MIN_WINDOW_HEIGHT = 600      # Minimum window height in pixels

# ============================================================================

# Import augmentation evaluator
try:
    from augmentation_evaluator import AugmentationEvaluator, export_to_excel
    AUGMENTATION_AVAILABLE = True
except ImportError:
    print("Warning: Augmentation evaluator not available. Install scipy, pandas, openpyxl.")
    AUGMENTATION_AVAILABLE = False


class VideoFilters:
    """
    Collection of 22 real-time video filters organized by category.
    All methods are static and work with OpenCV BGR images.
    """
    
    # ==================== EDGE DETECTION ====================
    
    @staticmethod
    def apply_sobel(frame: np.ndarray) -> np.ndarray:
        """
        Apply Sobel edge detection filter to the frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Filtered frame with Sobel edge detection applied
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel = np.uint8(np.clip(sobel, 0, 255))
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def apply_laplacian(frame: np.ndarray) -> np.ndarray:
        """
        Apply Laplacian filter to the frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Filtered frame with Laplacian applied
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))
        return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def apply_canny(frame: np.ndarray) -> np.ndarray:
        """
        Apply Canny edge detection filter.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Filtered frame with Canny edges
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # ==================== BLUR/SMOOTHING ====================
    
    @staticmethod
    def apply_gaussian_blur(frame: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur filter to the frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Filtered frame with Gaussian blur applied
        """
        return cv2.GaussianBlur(frame, (15, 15), 0)
    
    @staticmethod
    def apply_median_blur(frame: np.ndarray) -> np.ndarray:
        """
        Apply median blur filter (good for salt-and-pepper noise).
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Filtered frame with median blur applied
        """
        return cv2.medianBlur(frame, 15)
    
    @staticmethod
    def apply_bilateral(frame: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter (edge-preserving smoothing).
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Filtered frame with bilateral filter applied
        """
        return cv2.bilateralFilter(frame, 9, 75, 75)
    
    # ==================== ENHANCEMENT ====================
    
    @staticmethod
    def apply_sharpen(frame: np.ndarray) -> np.ndarray:
        """
        Apply sharpening filter using unsharp masking.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Sharpened frame
        """
        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(frame, -1, kernel)
    
    @staticmethod
    def apply_histogram_eq(frame: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization (enhances contrast).
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Contrast-enhanced frame
        """
        # Convert to YUV color space
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        
        # Equalize the histogram of the Y channel
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        
        # Convert back to BGR
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    # ==================== COLOR EFFECTS ====================
    
    @staticmethod
    def apply_grayscale(frame: np.ndarray) -> np.ndarray:
        """
        Convert to grayscale.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Grayscale frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def apply_sepia(frame: np.ndarray) -> np.ndarray:
        """
        Apply sepia tone filter (vintage/warm effect).
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Sepia-toned frame
        """
        # Sepia transformation matrix
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        
        sepia = cv2.transform(frame, kernel)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        return sepia
    
    @staticmethod
    def apply_invert(frame: np.ndarray) -> np.ndarray:
        """
        Invert colors (negative effect).
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Inverted frame
        """
        return cv2.bitwise_not(frame)
    
    @staticmethod
    def apply_warm(frame: np.ndarray) -> np.ndarray:
        """
        Apply warm color temperature.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Warm-toned frame
        """
        result = frame.copy()
        # Increase red and decrease blue
        result[:, :, 2] = np.clip(result[:, :, 2] * 1.2, 0, 255).astype(np.uint8)  # Red
        result[:, :, 0] = np.clip(result[:, :, 0] * 0.8, 0, 255).astype(np.uint8)  # Blue
        return result
    
    @staticmethod
    def apply_cool(frame: np.ndarray) -> np.ndarray:
        """
        Apply cool color temperature.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Cool-toned frame
        """
        result = frame.copy()
        # Increase blue and decrease red
        result[:, :, 0] = np.clip(result[:, :, 0] * 1.2, 0, 255).astype(np.uint8)  # Blue
        result[:, :, 2] = np.clip(result[:, :, 2] * 0.8, 0, 255).astype(np.uint8)  # Red
        return result
    
    # ==================== ARTISTIC EFFECTS ====================
    
    @staticmethod
    def apply_cartoon(frame: np.ndarray) -> np.ndarray:
        """
        Apply cartoon effect (bilateral filter + edge detection).
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Cartoonized frame
        """
        # Apply bilateral filter for color smoothing
        color = cv2.bilateralFilter(frame, 9, 250, 250)
        
        # Convert to grayscale and apply median blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        
        # Detect edges
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9, 2
        )
        
        # Convert edges to BGR
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine color and edges
        return cv2.bitwise_and(color, edges)
    
    @staticmethod
    def apply_sketch(frame: np.ndarray) -> np.ndarray:
        """
        Apply pencil sketch effect.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Sketch-style frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Invert the grayscale image
        inv_gray = 255 - gray
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        
        # Invert the blurred image
        inv_blur = 255 - blur
        
        # Create sketch by dividing
        sketch = cv2.divide(gray, inv_blur, scale=256.0)
        
        # Convert to BGR
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def apply_emboss(frame: np.ndarray) -> np.ndarray:
        """
        Apply emboss effect (3D raised appearance).
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Embossed frame
        """
        # Emboss kernel
        kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]])
        
        embossed = cv2.filter2D(frame, -1, kernel)
        
        # Add gray offset to make it visible
        embossed = cv2.add(embossed, np.array([128]))
        return embossed
    
    @staticmethod
    def apply_oil_painting(frame: np.ndarray) -> np.ndarray:
        """
        Apply oil painting effect.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Oil painting style frame
        """
        # Reduce image size for faster processing
        small = cv2.resize(frame, None, fx=0.5, fy=0.5)
        
        # Apply bilateral filter multiple times
        for _ in range(2):
            small = cv2.bilateralFilter(small, 9, 80, 80)
        
        # Resize back
        return cv2.resize(small, (frame.shape[1], frame.shape[0]))
    
    # ==================== MORPHOLOGICAL OPERATIONS ====================
    
    @staticmethod
    def apply_erosion(frame: np.ndarray) -> np.ndarray:
        """
        Apply morphological erosion.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Eroded frame
        """
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(frame, kernel, iterations=1)
    
    @staticmethod
    def apply_dilation(frame: np.ndarray) -> np.ndarray:
        """
        Apply morphological dilation.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dilated frame
        """
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(frame, kernel, iterations=1)
    
    # ==================== SPECIAL EFFECTS ====================
    
    @staticmethod
    def apply_pixelate(frame: np.ndarray) -> np.ndarray:
        """
        Apply pixelation effect.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Pixelated frame
        """
        # Get dimensions
        height, width = frame.shape[:2]
        
        # Downscale
        temp = cv2.resize(frame, (width // 15, height // 15), interpolation=cv2.INTER_LINEAR)
        
        # Upscale back using nearest neighbor
        return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    
    @staticmethod
    def apply_vignette(frame: np.ndarray) -> np.ndarray:
        """
        Apply vignette effect (darkened edges).
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Frame with vignette effect
        """
        rows, cols = frame.shape[:2]
        
        # Generate vignette mask
        X_resultant_kernel = cv2.getGaussianKernel(cols, cols / 2)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, rows / 2)
        
        resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = resultant_kernel / resultant_kernel.max()
        
        # Apply mask to each channel
        output = frame.copy()
        for i in range(3):
            output[:, :, i] = output[:, :, i] * mask
        
        return output.astype(np.uint8)
    
    @staticmethod
    def no_filter(frame: np.ndarray) -> np.ndarray:
        """
        Return the original frame without any filtering.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Original frame unchanged
        """
        return frame


class FeatureDetectors:
    """
    Computer vision feature detection algorithms with visual overlays.
    Includes HOG, ORB, SIFT, SURF, and Canny edge detection.
    
    Note: SIFT/SURF require opencv-contrib-python package.
    """
    
    @staticmethod
    def apply_hog(frame: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Apply HOG (Histogram of Oriented Gradients) feature detection.
        
        HOG is commonly used for object detection (e.g., pedestrian detection).
        Displays gradient orientations as visual overlay.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (frame with HOG visualization, info text)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Initialize HOG descriptor
            win_size = (64, 128)
            block_size = (16, 16)
            block_stride = (8, 8)
            cell_size = (8, 8)
            n_bins = 9
            
            hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)
            
            # Resize frame to match HOG window size for visualization
            h, w = gray.shape
            scale_factor = min(w // win_size[0], h // win_size[1])
            
            if scale_factor < 1:
                scale_factor = 1
            
            # Create visualization
            result_frame = frame.copy()
            
            # Compute HOG for visualization (cell-based gradient visualization)
            cell_size = 8
            for y in range(0, h - cell_size, cell_size):
                for x in range(0, w - cell_size, cell_size):
                    cell = gray[y:y+cell_size, x:x+cell_size]
                    
                    # Calculate gradient magnitude
                    gx = cv2.Sobel(cell, cv2.CV_32F, 1, 0, ksize=1)
                    gy = cv2.Sobel(cell, cv2.CV_32F, 0, 1, ksize=1)
                    mag = np.sqrt(gx**2 + gy**2).mean()
                    angle = np.arctan2(gy.mean(), gx.mean())
                    
                    # Draw orientation line if magnitude is significant
                    if mag > 10:
                        center = (x + cell_size // 2, y + cell_size // 2)
                        length = int(min(mag / 5, cell_size // 2))
                        end_x = int(center[0] + length * np.cos(angle))
                        end_y = int(center[1] + length * np.sin(angle))
                        
                        # Color: green for strong gradients
                        cv2.arrowedLine(result_frame, center, (end_x, end_y), 
                                      (0, 255, 0), 1, tipLength=0.3)
            
            # Add text overlay
            cv2.putText(result_frame, "HOG: Gradient Orientations", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            info = f"HOG Features: Gradient orientations displayed as green arrows"
            return result_frame, info
            
        except Exception as e:
            return frame, f"HOG Error: {str(e)}"
    
    @staticmethod
    def apply_orb(frame: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Apply ORB (Oriented FAST and Rotated BRIEF) feature detection.
        
        ORB is a fast, rotation-invariant feature detector.
        Displays keypoints as circles.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (frame with ORB keypoints, info text)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Create ORB detector
            orb = cv2.ORB_create(nfeatures=MAX_KEYPOINTS_ORB)
            
            # Detect keypoints and compute descriptors
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            # Draw keypoints on the frame
            result_frame = cv2.drawKeypoints(
                frame, keypoints, None, 
                color=(0, 255, 255),  # Cyan color for keypoints
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            
            # Add text overlay
            cv2.putText(result_frame, f"ORB: {len(keypoints)} keypoints", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            info = f"ORB Features: {len(keypoints)} keypoints detected (cyan circles)"
            return result_frame, info
            
        except Exception as e:
            return frame, f"ORB Error: {str(e)}"
    
    @staticmethod
    def apply_sift(frame: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Apply SIFT (Scale-Invariant Feature Transform) feature detection.
        
        SIFT detects and describes local features that are invariant to scale and rotation.
        Note: SIFT may require opencv-contrib-python package.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (frame with SIFT keypoints, info text)
        """
        try:
            # Check if SIFT is available
            if not hasattr(cv2, 'SIFT_create'):
                return frame, "SIFT not available. Install: pip install opencv-contrib-python"
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Create SIFT detector
            sift = cv2.SIFT_create(nfeatures=MAX_KEYPOINTS_SIFT)
            
            # Detect keypoints and compute descriptors
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            
            # Draw keypoints on the frame
            result_frame = cv2.drawKeypoints(
                frame, keypoints, None,
                color=(255, 0, 255),  # Magenta color for keypoints
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            
            # Add text overlay
            cv2.putText(result_frame, f"SIFT: {len(keypoints)} keypoints", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            info = f"SIFT Features: {len(keypoints)} scale-invariant keypoints (magenta)"
            return result_frame, info
            
        except AttributeError:
            return frame, "SIFT not available. Install: pip install opencv-contrib-python"
        except Exception as e:
            return frame, f"SIFT Error: {str(e)}"
    
    @staticmethod
    def apply_surf(frame: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Apply SURF (Speeded Up Robust Features) feature detection.
        
        SURF is a faster alternative to SIFT.
        Note: SURF is patented and requires opencv-contrib-python (non-free version).
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (frame with SURF keypoints, info text)
        """
        try:
            # Check if SURF is available
            if not hasattr(cv2, 'xfeatures2d'):
                return frame, "SURF not available. Requires opencv-contrib-python (non-free)"
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Create SURF detector
            surf = cv2.xfeatures2d.SURF_create(MAX_KEYPOINTS_SURF)
            
            # Detect keypoints and compute descriptors
            keypoints, descriptors = surf.detectAndCompute(gray, None)
            
            # Draw keypoints on the frame
            result_frame = cv2.drawKeypoints(
                frame, keypoints, None,
                color=(0, 165, 255),  # Orange color for keypoints
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            
            # Add text overlay
            cv2.putText(result_frame, f"SURF: {len(keypoints)} keypoints", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            info = f"SURF Features: {len(keypoints)} robust keypoints (orange)"
            return result_frame, info
            
        except AttributeError:
            return frame, "SURF not available. Requires opencv-contrib-python (non-free)"
        except Exception as e:
            return frame, f"SURF Error: {str(e)}"
    
    @staticmethod
    
    @staticmethod
    def apply_ferif(frame: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Apply FERIF (Fast Estimate Robust Image Features) detection.

        Uses the FERIFDetector to find robust local features and overlays them
        on the frame as keypoints.
        """
        try:
            detector = FERIFDetector()
            keypoints, descriptors = detector.detect_and_compute(frame)

            # Draw keypoints with rich visualization
            result_frame = cv2.drawKeypoints(
                frame,
                keypoints,
                None,
                color=(0, 215, 255),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

            info = (
                f"FERIF: {len(keypoints)} keypoints, "
                f"descriptor dim = {descriptors.shape[1] if descriptors is not None and len(descriptors) > 0 else 0}"
            )
            cv2.putText(
                result_frame,
                "FERIF: Max-Info Features",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 215, 255),
                2,
            )
            return result_frame, info
        except Exception as e:
            return frame, f"FERIF Error: {str(e)}"

    @staticmethod
    def apply_canny_edges(frame: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Apply Canny edge detection (bonus feature).
        
        Detects edges with hysteresis thresholding.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (frame with edges overlay, info text)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
            
            # Detect edges
            edges = cv2.Canny(blurred, 50, 150)
            
            # Create colored overlay (red edges on original frame)
            result_frame = frame.copy()
            result_frame[edges != 0] = [0, 0, 255]  # Red color for edges
            
            # Blend with original
            result_frame = cv2.addWeighted(frame, 0.7, result_frame, 0.3, 0)
            
            # Add text overlay
            edge_count = np.count_nonzero(edges)
            cv2.putText(result_frame, f"Canny Edges: {edge_count} pixels", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            info = f"Canny Edge Detection: {edge_count} edge pixels (red overlay)"
            return result_frame, info
            
        except Exception as e:
            return frame, f"Canny Error: {str(e)}"


class SavedFrame:
    """Data container for a saved frame with its metadata."""
    
    def __init__(self, frame: np.ndarray, frame_index: int, timestamp: str):
        self.frame = frame.copy()
        self.frame_index = frame_index
        self.timestamp = timestamp
        self.thumbnail: Optional[ImageTk.PhotoImage] = None
        self.id = id(self)  # Unique identifier


class FrameGallery:
    """Manages the collection of saved frames."""
    
    def __init__(self):
        self.frames: List[SavedFrame] = []
        self.max_frames = MAX_SAVED_FRAMES
    
    def add_frame(self, frame: np.ndarray, frame_index: int = 0) -> SavedFrame:
        """
        Add a frame to the gallery.
        
        Args:
            frame: Frame to save
            frame_index: Frame index in the video
            
        Returns:
            The SavedFrame object
        """
        # Limit number of saved frames
        if len(self.frames) >= self.max_frames:
            self.frames.pop(0)  # Remove oldest
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        saved_frame = SavedFrame(frame, frame_index, timestamp)
        self.frames.append(saved_frame)
        
        return saved_frame
    
    def get_frame(self, index: int) -> Optional[SavedFrame]:
        """
        Get a frame by index.
        
        Args:
            index: Index in the gallery
            
        Returns:
            SavedFrame or None
        """
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return None
    
    def get_frame_count(self) -> int:
        """Get the number of saved frames."""
        return len(self.frames)
    
    def clear(self):
        """Clear all saved frames."""
        self.frames.clear()


class VideoManager:
    """Handles video file and webcam capture using OpenCV."""
    
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_playing = False
        self.current_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self.fps = 0
        self.current_frame_index = 0
        self.video_source = None
        
    def open_video(self, filepath: str) -> bool:
        """
        Open a video file for playback.
        
        Args:
            filepath: Path to the video file
            
        Returns:
            True if video opened successfully, False otherwise
        """
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(filepath)
        
        if not self.cap.isOpened():
            return False
        
        self.video_source = filepath
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame_index = 0
        
        # Read the first frame
        ret, self.current_frame = self.cap.read()
        
        return ret
    
    def open_webcam(self, camera_index: int = 0) -> bool:
        """
        Open webcam for live video capture.
        
        Args:
            camera_index: Index of the camera (default: 0)
            
        Returns:
            True if webcam opened successfully, False otherwise
        """
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            return False
        
        self.video_source = f"webcam_{camera_index}"
        self.frame_count = -1  # Infinite for webcam
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if not available
        self.current_frame_index = 0
        
        # Read the first frame
        ret, self.current_frame = self.cap.read()
        
        return ret
    
    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the video source.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame = frame
            self.current_frame_index += 1
        
        return ret, frame
    
    def set_frame_position(self, frame_index: int) -> bool:
        """
        Set the video position to a specific frame.
        
        Args:
            frame_index: Frame index to jump to
            
        Returns:
            True if successful, False otherwise
        """
        if self.cap is None or self.is_webcam():
            return False
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame = frame
            self.current_frame_index = frame_index
        
        return ret
    
    def is_webcam(self) -> bool:
        """Check if current source is a webcam."""
        return self.video_source is not None and "webcam" in str(self.video_source)
    
    def get_frame_delay(self) -> int:
        """
        Calculate the delay between frames in milliseconds.
        
        Returns:
            Delay in milliseconds
        """
        if self.fps > 0:
            return int(1000 / self.fps)
        return 33  # Default ~30 FPS
    
    def release(self):
        """Release the video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VideoFilterApp:
    """Main GUI application for video filtering and feature detection."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Video Filter & Feature Detection Application")
        
        # Make window responsive to screen size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Use 90% of screen size, but with min/max limits
        window_width = min(max(1200, int(screen_width * 0.9)), screen_width - 100)
        window_height = min(max(700, int(screen_height * 0.85)), screen_height - 100)
        
        # Center the window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Set minimum window size
        self.root.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        
        # Allow window resizing
        self.root.resizable(True, True)
        
        # Initialize components
        self.video_manager = VideoManager()
        self.filters = VideoFilters()
        self.feature_detectors = FeatureDetectors()
        self.gallery = FrameGallery()
        
        # Filter state (for combining multiple filters)
        self.filter_states = {
            # Edge Detection
            'sobel': tk.BooleanVar(value=False),
            'laplacian': tk.BooleanVar(value=False),
            'canny': tk.BooleanVar(value=False),
            # Blur/Smoothing
            'gaussian': tk.BooleanVar(value=False),
            'median': tk.BooleanVar(value=False),
            'bilateral': tk.BooleanVar(value=False),
            # Enhancement
            'sharpen': tk.BooleanVar(value=False),
            'hist_eq': tk.BooleanVar(value=False),
            # Color Effects
            'grayscale': tk.BooleanVar(value=False),
            'sepia': tk.BooleanVar(value=False),
            'invert': tk.BooleanVar(value=False),
            'warm': tk.BooleanVar(value=False),
            'cool': tk.BooleanVar(value=False),
            # Artistic
            'cartoon': tk.BooleanVar(value=False),
            'sketch': tk.BooleanVar(value=False),
            'emboss': tk.BooleanVar(value=False),
            'oil_paint': tk.BooleanVar(value=False),
            # Morphological
            'erosion': tk.BooleanVar(value=False),
            'dilation': tk.BooleanVar(value=False),
            # Special Effects
            'pixelate': tk.BooleanVar(value=False),
            'vignette': tk.BooleanVar(value=False)
        }
        
        # Feature detection state
        self.selected_gallery_index: Optional[int] = None
        self.current_feature_overlay = tk.StringVar(value="none")
        self.feature_info_text = ""
        
        # Threading control
        self.playback_thread: Optional[threading.Thread] = None
        self.stop_playback_flag = threading.Event()
        
        # Prevent circular slider updates
        self.updating_slider = False
        
        # UI components
        self.video_label: Optional[tk.Label] = None
        self.progress_slider: Optional[ttk.Scale] = None
        self.frame_label: Optional[tk.Label] = None
        self.gallery_canvas: Optional[tk.Canvas] = None
        self.gallery_frame: Optional[tk.Frame] = None
        self.feature_info_label: Optional[tk.Label] = None
        
        # Build UI
        self.create_ui()
        
        # Bind cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_ui(self):
        """Create the user interface components."""
        # Main container with three sections: video (left), gallery (right)
        main_container = tk.Frame(self.root, bg="#1a1a1a")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel: Video player
        left_panel = tk.Frame(main_container, bg="#1a1a1a")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel: Gallery
        right_panel = tk.Frame(main_container, bg="#2c3e50", width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        right_panel.pack_propagate(False)
        
        # === LEFT PANEL: VIDEO SECTION ===
        
        # Top frame for file selection (fixed height)
        top_frame = tk.Frame(left_panel, bg="#2c3e50", height=60)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        top_frame.pack_propagate(False)
        
        tk.Button(
            top_frame,
            text="Open Video File",
            command=self.open_video_file,
            bg="#3498db",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            top_frame,
            text="Open Webcam",
            command=self.open_webcam,
            bg="#9b59b6",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5
        ).pack(side=tk.LEFT, padx=5)
        
        # Video display area (responsive size)
        video_frame = tk.Frame(left_panel, bg="black")
        video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_label = tk.Label(video_frame, bg="black")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")  # Center the image
        
        # Store video frame reference for responsive sizing
        self.video_frame = video_frame
        
        # Progress slider frame (fixed height)
        progress_frame = tk.Frame(left_panel, bg="#34495e", height=50)
        progress_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        progress_frame.pack_propagate(False)
        
        self.frame_label = tk.Label(
            progress_frame,
            text="Frame: 0 / 0",
            bg="#34495e",
            fg="white",
            font=("Arial", 9)
        )
        self.frame_label.pack(side=tk.TOP, pady=2)
        
        self.progress_slider = ttk.Scale(
            progress_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            command=self.on_slider_change
        )
        self.progress_slider.pack(side=tk.TOP, fill=tk.X, padx=10, pady=2)
        
        # Control buttons frame (fixed height)
        control_frame = tk.Frame(left_panel, bg="#2c3e50", height=60)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        control_frame.pack_propagate(False)
        
        tk.Button(
            control_frame,
            text="⏮ Previous",
            command=self.previous_frame,
            bg="#95a5a6",
            fg="white",
            font=("Arial", 9),
            padx=10,
            pady=5
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            control_frame,
            text="▶ Play",
            command=self.play_video,
            bg="#27ae60",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=5
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            control_frame,
            text="⏸ Pause",
            command=self.pause_video,
            bg="#e67e22",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=5
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            control_frame,
            text="⏭ Next",
            command=self.next_frame,
            bg="#95a5a6",
            fg="white",
            font=("Arial", 9),
            padx=10,
            pady=5
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            control_frame,
            text="💾 Save Frame",
            command=self.save_current_frame,
            bg="#16a085",
            fg="white",
            font=("Arial", 9, "bold"),
            padx=10,
            pady=5
        ).pack(side=tk.RIGHT, padx=5)
        
        # Filter selection frame (fixed size to prevent UI leaking)
        filter_container = tk.Frame(left_panel, bg="#34495e", height=150)
        filter_container.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        filter_container.pack_propagate(False)  # Prevent auto-resizing
        
        # Header with title and clear button
        filter_header = tk.Frame(filter_container, bg="#34495e")
        filter_header.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        tk.Label(
            filter_header,
            text="🎨 Real-Time Filters (Check multiple to combine):",
            bg="#34495e",
            fg="white",
            font=("Arial", 10, "bold")
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            filter_header,
            text="Clear All",
            command=self.clear_all_filters,
            bg="#c0392b",
            fg="white",
            font=("Arial", 8),
            padx=8,
            pady=3
        ).pack(side=tk.RIGHT, padx=5)
        
        # Create scrollable filter grid with categories
        filter_canvas = tk.Canvas(filter_container, bg="#34495e", height=120, highlightthickness=0)
        filter_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=2)
        
        filter_scrollbar = tk.Scrollbar(filter_container, orient=tk.HORIZONTAL, command=filter_canvas.xview)
        filter_scrollbar.pack(side=tk.BOTTOM, fill=tk.X, padx=5)
        filter_canvas.configure(xscrollcommand=filter_scrollbar.set)
        
        filter_grid = tk.Frame(filter_canvas, bg="#34495e")
        filter_canvas.create_window((0, 0), window=filter_grid, anchor="nw")
        
        # Bind to update scroll region
        def update_scroll_region(event=None):
            filter_canvas.configure(scrollregion=filter_canvas.bbox("all"))
        
        filter_grid.bind("<Configure>", update_scroll_region)
        
        # Enable mousewheel scrolling (horizontal)
        def on_mousewheel(event):
            # Shift+wheel for horizontal scroll on Windows
            filter_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        
        # Bind mousewheel events
        filter_canvas.bind("<MouseWheel>", on_mousewheel)
        filter_canvas.bind("<Shift-MouseWheel>", on_mousewheel)
        
        # Also bind to filter_grid so it works when hovering over checkboxes
        filter_grid.bind("<MouseWheel>", on_mousewheel)
        filter_grid.bind("<Shift-MouseWheel>", on_mousewheel)
        
        # Define filter categories and their filters
        filter_categories = [
            ("Edge Detection", [
                ('sobel', 'Sobel'),
                ('laplacian', 'Laplacian'),
                ('canny', 'Canny')
            ]),
            ("Blur/Smooth", [
                ('gaussian', 'Gaussian'),
                ('median', 'Median'),
                ('bilateral', 'Bilateral')
            ]),
            ("Enhancement", [
                ('sharpen', 'Sharpen'),
                ('hist_eq', 'Contrast')
            ]),
            ("Color Effects", [
                ('grayscale', 'Grayscale'),
                ('sepia', 'Sepia'),
                ('invert', 'Invert'),
                ('warm', 'Warm'),
                ('cool', 'Cool')
            ]),
            ("Artistic", [
                ('cartoon', 'Cartoon'),
                ('sketch', 'Sketch'),
                ('emboss', 'Emboss'),
                ('oil_paint', 'Oil Paint')
            ]),
            ("Morphology", [
                ('erosion', 'Erosion'),
                ('dilation', 'Dilation')
            ]),
            ("Special FX", [
                ('pixelate', 'Pixelate'),
                ('vignette', 'Vignette')
            ])
        ]
        
        # Create columns for filter categories (fixed width to prevent expansion)
        for col_idx, (category_name, filters) in enumerate(filter_categories):
            category_frame = tk.Frame(filter_grid, bg="#2c3e50", relief=tk.RAISED, 
                                     borderwidth=1, width=100)
            category_frame.grid(row=0, column=col_idx, padx=2, pady=2, sticky="nsew")
            category_frame.grid_propagate(False)  # Prevent auto-resizing
            
            # Inner frame for content
            inner_frame = tk.Frame(category_frame, bg="#2c3e50")
            inner_frame.pack(fill=tk.BOTH, expand=True)
            
            # Category label
            tk.Label(
                inner_frame,
                text=category_name,
                bg="#2c3e50",
                fg="#3498db",
                font=("Arial", 8, "bold"),
                wraplength=90
            ).pack(side=tk.TOP, pady=1)
            
            # Filter checkboxes
            for filter_key, filter_label in filters:
                cb = tk.Checkbutton(
                    inner_frame,
                    text=filter_label,
                    variable=self.filter_states[filter_key],
                    command=self.on_filter_change,
                    bg="#2c3e50",
                    fg="white",
                    selectcolor="#34495e",
                    font=("Arial", 7),
                    activebackground="#2c3e50",
                    activeforeground="white",
                    anchor="w",
                    wraplength=90
                )
                cb.pack(side=tk.TOP, fill=tk.X, padx=3, pady=0)
        
        # === RIGHT PANEL: GALLERY SECTION ===
        
        # Gallery header
        gallery_header = tk.Frame(right_panel, bg="#2c3e50")
        gallery_header.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        tk.Label(
            gallery_header,
            text="📸 Saved Frames Gallery",
            bg="#2c3e50",
            fg="white",
            font=("Arial", 12, "bold")
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            gallery_header,
            text="Clear All",
            command=self.clear_gallery,
            bg="#c0392b",
            fg="white",
            font=("Arial", 8),
            padx=8,
            pady=3
        ).pack(side=tk.RIGHT, padx=5)
        
        # Gallery scrollable area
        gallery_container = tk.Frame(right_panel, bg="#34495e")
        gallery_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas with scrollbar
        gallery_scrollbar = tk.Scrollbar(gallery_container, orient=tk.VERTICAL)
        gallery_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.gallery_canvas = tk.Canvas(
            gallery_container,
            bg="#34495e",
            yscrollcommand=gallery_scrollbar.set,
            highlightthickness=0
        )
        self.gallery_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        gallery_scrollbar.config(command=self.gallery_canvas.yview)
        
        # Frame inside canvas for gallery items
        self.gallery_frame = tk.Frame(self.gallery_canvas, bg="#34495e")
        self.gallery_canvas.create_window((0, 0), window=self.gallery_frame, anchor="nw")
        
        # Bind canvas resize
        self.gallery_frame.bind("<Configure>", 
                               lambda e: self.gallery_canvas.configure(
                                   scrollregion=self.gallery_canvas.bbox("all")
                               ))
        
        # Feature detection panel
        feature_panel = tk.Frame(right_panel, bg="#2c3e50")
        feature_panel.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        tk.Label(
            feature_panel,
            text="🔍 Feature Detection",
            bg="#2c3e50",
            fg="white",
            font=("Arial", 11, "bold")
        ).pack(side=tk.TOP, anchor="w", padx=5, pady=5)
        
        tk.Label(
            feature_panel,
            text="Select algorithm:",
            bg="#2c3e50",
            fg="white",
            font=("Arial", 9)
        ).pack(side=tk.TOP, anchor="w", padx=5)
        
        # Feature detection radio buttons
        features = [
            ("None", "none"),
            ("HOG (Gradients)", "hog"),
            ("ORB (Keypoints)", "orb"),
            ("SIFT (Scale-Inv.)", "sift"),
            ("SURF (Fast)", "surf"),
            ("FERIF (Max-Info)", "ferif"),
            ("Canny Edges", "canny")
        ]
        
        for feature_name, feature_value in features:
            rb = tk.Radiobutton(
                feature_panel,
                text=feature_name,
                variable=self.current_feature_overlay,
                value=feature_value,
                command=self.apply_feature_overlay,
                bg="#2c3e50",
                fg="white",
                selectcolor="#34495e",
                font=("Arial", 9),
                activebackground="#2c3e50",
                activeforeground="white"
            )
            rb.pack(side=tk.TOP, anchor="w", padx=20, pady=2)
        
        # Feature info display
        self.feature_info_label = tk.Label(
            feature_panel,
            text="Select a saved frame to apply feature detection",
            bg="#34495e",
            fg="#ecf0f1",
            font=("Arial", 8),
            wraplength=300,
            justify=tk.LEFT,
            padx=5,
            pady=5
        )
        self.feature_info_label.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Augmentation evaluation button
        if AUGMENTATION_AVAILABLE:
            tk.Button(
                feature_panel,
                text="📊 Evaluate Augmentation Robustness",
                command=self.open_augmentation_evaluation,
                bg="#8e44ad",
                fg="white",
                font=("Arial", 9, "bold"),
                padx=10,
                pady=5
            ).pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)
    
    def open_video_file(self):
        """Open a video file dialog and load the selected video."""
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                ("All Files", "*.*")
            ]
        )
        
        if filepath:
            self.pause_video()
            
            if self.video_manager.open_video(filepath):
                self.progress_slider.config(to=self.video_manager.frame_count - 1)
                self.update_frame_display()
                messagebox.showinfo("Success", "Video loaded successfully!")
            else:
                messagebox.showerror("Error", "Failed to open video file.")
    
    def open_webcam(self):
        """Open the default webcam."""
        self.pause_video()
        
        if self.video_manager.open_webcam():
            self.progress_slider.config(to=0)  # Disable slider for webcam
            self.update_frame_display()
            messagebox.showinfo("Success", "Webcam opened successfully!")
            # Auto-start playback for webcam
            self.play_video()
        else:
            messagebox.showerror("Error", "Failed to open webcam.")
    
    def on_filter_change(self):
        """
        Handle filter checkbox changes.
        Update display immediately if not playing.
        """
        # Update display immediately if not playing
        if not self.video_manager.is_playing:
            self.update_frame_display()
    
    def clear_all_filters(self):
        """Clear all selected filters."""
        for state in self.filter_states.values():
            state.set(False)
        
        # Update display immediately if not playing
        if not self.video_manager.is_playing:
            self.update_frame_display()
    
    def apply_combined_filters(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply all selected filters to the frame in a logical sequence.
        
        Filter application order:
        1. Blur/Smoothing (noise reduction)
        2. Color adjustments
        3. Enhancement
        4. Edge detection
        5. Artistic effects
        6. Morphological operations
        7. Special effects
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Frame with all selected filters applied
        """
        filtered_frame = frame.copy()
        
        # Step 1: Blur/Smoothing (apply before other processing)
        if self.filter_states['gaussian'].get():
            filtered_frame = self.filters.apply_gaussian_blur(filtered_frame)
        
        if self.filter_states['median'].get():
            filtered_frame = self.filters.apply_median_blur(filtered_frame)
        
        if self.filter_states['bilateral'].get():
            filtered_frame = self.filters.apply_bilateral(filtered_frame)
        
        # Step 2: Color adjustments
        if self.filter_states['grayscale'].get():
            filtered_frame = self.filters.apply_grayscale(filtered_frame)
        
        if self.filter_states['sepia'].get():
            filtered_frame = self.filters.apply_sepia(filtered_frame)
        
        if self.filter_states['warm'].get():
            filtered_frame = self.filters.apply_warm(filtered_frame)
        
        if self.filter_states['cool'].get():
            filtered_frame = self.filters.apply_cool(filtered_frame)
        
        if self.filter_states['invert'].get():
            filtered_frame = self.filters.apply_invert(filtered_frame)
        
        # Step 3: Enhancement
        if self.filter_states['sharpen'].get():
            filtered_frame = self.filters.apply_sharpen(filtered_frame)
        
        if self.filter_states['hist_eq'].get():
            filtered_frame = self.filters.apply_histogram_eq(filtered_frame)
        
        # Step 4: Edge detection
        if self.filter_states['sobel'].get():
            filtered_frame = self.filters.apply_sobel(filtered_frame)
        
        if self.filter_states['laplacian'].get():
            filtered_frame = self.filters.apply_laplacian(filtered_frame)
        
        if self.filter_states['canny'].get():
            filtered_frame = self.filters.apply_canny(filtered_frame)
        
        # Step 5: Artistic effects
        if self.filter_states['cartoon'].get():
            filtered_frame = self.filters.apply_cartoon(filtered_frame)
        
        if self.filter_states['sketch'].get():
            filtered_frame = self.filters.apply_sketch(filtered_frame)
        
        if self.filter_states['emboss'].get():
            filtered_frame = self.filters.apply_emboss(filtered_frame)
        
        if self.filter_states['oil_paint'].get():
            filtered_frame = self.filters.apply_oil_painting(filtered_frame)
        
        # Step 6: Morphological operations
        if self.filter_states['erosion'].get():
            filtered_frame = self.filters.apply_erosion(filtered_frame)
        
        if self.filter_states['dilation'].get():
            filtered_frame = self.filters.apply_dilation(filtered_frame)
        
        # Step 7: Special effects
        if self.filter_states['pixelate'].get():
            filtered_frame = self.filters.apply_pixelate(filtered_frame)
        
        if self.filter_states['vignette'].get():
            filtered_frame = self.filters.apply_vignette(filtered_frame)
        
        return filtered_frame
    
    def play_video(self):
        """Start video playback in a separate thread."""
        if self.video_manager.cap is None:
            messagebox.showwarning("Warning", "Please open a video file or webcam first.")
            return
        
        if self.video_manager.is_playing:
            return  # Already playing
        
        self.video_manager.is_playing = True
        self.stop_playback_flag.clear()
        
        # Start playback thread
        self.playback_thread = threading.Thread(target=self.playback_loop, daemon=True)
        self.playback_thread.start()
    
    def pause_video(self):
        """Pause video playback."""
        self.video_manager.is_playing = False
        self.stop_playback_flag.set()
        
        # Wait for playback thread to finish
        if self.playback_thread is not None and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)
    
    def playback_loop(self):
        """
        Main playback loop running in a separate thread.
        Continuously reads frames and updates the display.
        """
        frame_delay = self.video_manager.get_frame_delay() / 1000.0  # Convert to seconds
        
        while self.video_manager.is_playing and not self.stop_playback_flag.is_set():
            start_time = time.time()
            
            ret, frame = self.video_manager.read_frame()
            
            if not ret:
                # End of video or error
                if not self.video_manager.is_webcam():
                    # Loop video for files
                    self.video_manager.set_frame_position(0)
                else:
                    break
            else:
                # Update display in main thread
                self.root.after(0, self.update_frame_display)
            
            # Maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_delay - elapsed)
            time.sleep(sleep_time)
        
        self.video_manager.is_playing = False
    
    def next_frame(self):
        """Move to the next frame."""
        if self.video_manager.cap is None or self.video_manager.is_webcam():
            return
        
        if self.video_manager.is_playing:
            self.pause_video()
        
        current_pos = self.video_manager.current_frame_index
        if current_pos < self.video_manager.frame_count - 1:
            self.video_manager.set_frame_position(current_pos + 1)
            self.update_frame_display()
    
    def previous_frame(self):
        """Move to the previous frame."""
        if self.video_manager.cap is None or self.video_manager.is_webcam():
            return
        
        if self.video_manager.is_playing:
            self.pause_video()
        
        current_pos = self.video_manager.current_frame_index
        if current_pos > 0:
            self.video_manager.set_frame_position(current_pos - 1)
            self.update_frame_display()
    
    def on_slider_change(self, value):
        """
        Handle slider position change.
        
        Args:
            value: New slider value
        """
        # Prevent circular updates
        if self.updating_slider:
            return
        
        if self.video_manager.cap is None or self.video_manager.is_webcam():
            return
        
        if self.video_manager.is_playing:
            return  # Don't respond to slider during playback
        
        frame_index = int(float(value))
        self.video_manager.set_frame_position(frame_index)
        self.update_frame_display()
    
    def update_frame_display(self):
        """Update the video frame display with all selected filters applied."""
        if self.video_manager.current_frame is None:
            return
        
        # Apply combined filters
        filtered_frame = self.apply_combined_filters(self.video_manager.current_frame)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Get container dimensions (use parent frame, not label which can grow)
        container = self.video_label.master
        display_width = container.winfo_width()
        display_height = container.winfo_height()
        
        # Use responsive size if container not yet sized
        if display_width <= 1 or display_height <= 1:
            # Use percentage of window size
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()
            display_width = max(600, int(window_width * 0.6))
            display_height = max(400, int(window_height * 0.6))
        
        # Resize to fit display area while maintaining aspect ratio
        # Use a slightly smaller size to ensure no overflow
        max_width = display_width - 10
        max_height = display_height - 10
        
        # Calculate scaling to maintain aspect ratio
        img_width, img_height = pil_image.size
        width_ratio = max_width / img_width
        height_ratio = max_height / img_height
        scale = min(width_ratio, height_ratio)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image to exact dimensions (not thumbnail which only limits max)
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to ImageTk
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update label
        self.video_label.config(image=photo)
        self.video_label.image = photo  # Keep a reference
        
        # Update progress slider and frame label
        if not self.video_manager.is_webcam():
            # Prevent circular callback when updating slider programmatically
            self.updating_slider = True
            try:
                self.progress_slider.set(self.video_manager.current_frame_index)
                self.frame_label.config(
                    text=f"Frame: {self.video_manager.current_frame_index} / {self.video_manager.frame_count}"
                )
            finally:
                self.updating_slider = False
        else:
            self.frame_label.config(text=f"Webcam (Live)")
    
    def save_current_frame(self):
        """Save the current frame to the gallery."""
        if self.video_manager.current_frame is None:
            messagebox.showwarning("Warning", "No frame to save.")
            return
        
        # Apply current filters
        filtered_frame = self.apply_combined_filters(self.video_manager.current_frame)
        
        # Add to gallery
        saved_frame = self.gallery.add_frame(
            filtered_frame,
            self.video_manager.current_frame_index
        )
        
        # Update gallery display
        self.refresh_gallery()
        
        messagebox.showinfo("Success", f"Frame saved to gallery!\nTotal: {self.gallery.get_frame_count()}")
    
    def refresh_gallery(self):
        """Refresh the gallery display with all saved frames."""
        # Clear existing gallery items
        for widget in self.gallery_frame.winfo_children():
            widget.destroy()
        
        # Add each saved frame as a thumbnail
        for index, saved_frame in enumerate(self.gallery.frames):
            self.create_gallery_item(saved_frame, index)
        
        # Update scroll region
        self.gallery_frame.update_idletasks()
        self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))
    
    def create_gallery_item(self, saved_frame: SavedFrame, index: int):
        """
        Create a gallery item widget.
        
        Args:
            saved_frame: The SavedFrame object
            index: Index in the gallery
        """
        # Container for this gallery item
        item_frame = tk.Frame(self.gallery_frame, bg="#2c3e50", relief=tk.RAISED, borderwidth=2)
        item_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Create thumbnail
        thumbnail_size = (300, 200)
        frame_rgb = cv2.cvtColor(saved_frame.frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        pil_image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Store reference
        saved_frame.thumbnail = photo
        
        # Thumbnail label (clickable)
        thumb_label = tk.Label(item_frame, image=photo, bg="#34495e", cursor="hand2")
        thumb_label.image = photo
        thumb_label.pack(side=tk.TOP, padx=5, pady=5)
        
        # Bind click event
        thumb_label.bind("<Button-1>", lambda e, idx=index: self.select_gallery_frame(idx))
        
        # Info frame
        info_frame = tk.Frame(item_frame, bg="#2c3e50")
        info_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        tk.Label(
            info_frame,
            text=f"Frame #{index + 1} (Video: {saved_frame.frame_index})",
            bg="#2c3e50",
            fg="white",
            font=("Arial", 9, "bold")
        ).pack(side=tk.TOP, anchor="w")
        
        tk.Label(
            info_frame,
            text=f"Saved: {saved_frame.timestamp}",
            bg="#2c3e50",
            fg="#bdc3c7",
            font=("Arial", 8)
        ).pack(side=tk.TOP, anchor="w")
        
        # Button frame
        button_frame = tk.Frame(item_frame, bg="#2c3e50")
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        tk.Button(
            button_frame,
            text="Select",
            command=lambda idx=index: self.select_gallery_frame(idx),
            bg="#3498db",
            fg="white",
            font=("Arial", 8),
            padx=10,
            pady=3
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            button_frame,
            text="Export",
            command=lambda idx=index: self.export_gallery_frame(idx),
            bg="#16a085",
            fg="white",
            font=("Arial", 8),
            padx=10,
            pady=3
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            button_frame,
            text="Delete",
            command=lambda idx=index: self.delete_gallery_frame(idx),
            bg="#c0392b",
            fg="white",
            font=("Arial", 8),
            padx=10,
            pady=3
        ).pack(side=tk.LEFT, padx=2)
    
    def select_gallery_frame(self, index: int):
        """
        Select a frame from the gallery for feature detection.
        
        Args:
            index: Index of the frame in the gallery
        """
        saved_frame = self.gallery.get_frame(index)
        if saved_frame is None:
            return
        
        self.selected_gallery_index = index
        
        # Apply feature overlay if one is selected
        self.apply_feature_overlay()
        
        # Update UI feedback
        self.feature_info_label.config(
            text=f"Selected Frame #{index + 1}. Choose a feature detection algorithm."
        )
    
    def apply_feature_overlay(self):
        """Apply the selected feature detection overlay to the selected gallery frame."""
        if self.selected_gallery_index is None:
            self.feature_info_label.config(
                text="No frame selected. Click a frame in the gallery to select it."
            )
            return
        
        saved_frame = self.gallery.get_frame(self.selected_gallery_index)
        if saved_frame is None:
            return
        
        overlay_type = self.current_feature_overlay.get()
        
        # Get the frame
        frame = saved_frame.frame.copy()
        info_text = ""
        
        # Apply selected feature detection
        if overlay_type == "hog":
            frame, info_text = self.feature_detectors.apply_hog(frame)
        elif overlay_type == "orb":
            frame, info_text = self.feature_detectors.apply_orb(frame)
        elif overlay_type == "sift":
            frame, info_text = self.feature_detectors.apply_sift(frame)
        elif overlay_type == "surf":
            frame, info_text = self.feature_detectors.apply_surf(frame)
        elif overlay_type == "ferif":
            frame, info_text = self.feature_detectors.apply_ferif(frame)
        elif overlay_type == "canny":
            frame, info_text = self.feature_detectors.apply_canny_edges(frame)
        else:
            info_text = f"Viewing Frame #{self.selected_gallery_index + 1} (no overlay)"
        
        # Display the result in the main video display area
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Get container dimensions (use parent frame, not label which can grow)
        container = self.video_label.master
        display_width = container.winfo_width()
        display_height = container.winfo_height()
        
        # Use responsive size if container not yet sized
        if display_width <= 1 or display_height <= 1:
            # Use percentage of window size
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()
            display_width = max(600, int(window_width * 0.6))
            display_height = max(400, int(window_height * 0.6))
        
        # Resize to fit display area while maintaining aspect ratio
        # Use a slightly smaller size to ensure no overflow
        max_width = display_width - 10
        max_height = display_height - 10
        
        # Calculate scaling to maintain aspect ratio
        img_width, img_height = pil_image.size
        width_ratio = max_width / img_width
        height_ratio = max_height / img_height
        scale = min(width_ratio, height_ratio)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image to exact dimensions (not thumbnail which only limits max)
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(pil_image)
        
        self.video_label.config(image=photo)
        self.video_label.image = photo
        
        # Update info label
        self.feature_info_label.config(text=info_text)
    
    def export_gallery_frame(self, index: int):
        """
        Export a gallery frame to a file.
        
        Args:
            index: Index of the frame in the gallery
        """
        saved_frame = self.gallery.get_frame(index)
        if saved_frame is None:
            return
        
        # Ask for save location
        filepath = filedialog.asksaveasfilename(
            title="Export Frame",
            defaultextension=".png",
            filetypes=[
                ("PNG Files", "*.png"),
                ("JPEG Files", "*.jpg"),
                ("All Files", "*.*")
            ]
        )
        
        if filepath:
            cv2.imwrite(filepath, saved_frame.frame)
            messagebox.showinfo("Success", f"Frame exported to:\n{filepath}")
    
    def delete_gallery_frame(self, index: int):
        """
        Delete a frame from the gallery.
        
        Args:
            index: Index of the frame in the gallery
        """
        if 0 <= index < len(self.gallery.frames):
            self.gallery.frames.pop(index)
            self.refresh_gallery()
            
            # Clear selection if deleted
            if self.selected_gallery_index == index:
                self.selected_gallery_index = None
                self.feature_info_label.config(text="Frame deleted. Select another frame.")
    
    def clear_gallery(self):
        """Clear all frames from the gallery."""
        if self.gallery.get_frame_count() == 0:
            return
        
        if messagebox.askyesno("Clear Gallery", "Delete all saved frames?"):
            self.gallery.clear()
            self.refresh_gallery()
            self.selected_gallery_index = None
            self.feature_info_label.config(text="Gallery cleared.")
    
    def open_augmentation_evaluation(self):
        """Open augmentation evaluation window for the selected frame."""
        if not AUGMENTATION_AVAILABLE:
            messagebox.showerror("Error", "Augmentation evaluation not available.\nInstall: pip install scipy pandas openpyxl")
            return
        
        if self.selected_gallery_index is None:
            messagebox.showwarning("Warning", "Please select a frame from the gallery first.")
            return
        
        saved_frame = self.gallery.get_frame(self.selected_gallery_index)
        if saved_frame is None:
            return
        
        # Ask for detector type
        detector_choice = self.current_feature_overlay.get()
        if detector_choice == "none":
            messagebox.showinfo("Info", "Please select a feature detector first (ORB, SIFT, SURF).")
            return
        
        # Check if detector is suitable for augmentation evaluation
        if detector_choice in ["hog", "canny"]:
            messagebox.showerror(
                "Incompatible Detector", 
                f"{detector_choice.upper()} is not suitable for augmentation evaluation.\n\n"
                f"Reason: {detector_choice.upper()} produces edge pixels or gradient histograms, "
                f"not matchable keypoint descriptors.\n\n"
                f"Please use: ORB, SIFT, or SURF\n\n"
                f"These detectors produce keypoints with descriptors that can be matched "
                f"across transformed images."
            )
            return
        
        # Create evaluation window
        eval_window = tk.Toplevel(self.root)
        eval_window.title(f"Augmentation Evaluation - {detector_choice.upper()}")
        eval_window.geometry("1000x600")
        eval_window.configure(bg="#2c3e50")
        
        # Header
        header_frame = tk.Frame(eval_window, bg="#34495e", height=60)
        header_frame.pack(side=tk.TOP, fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame,
            text=f"🔬 Augmentation Robustness Evaluation - {detector_choice.upper()}",
            bg="#34495e",
            fg="white",
            font=("Arial", 14, "bold")
        ).pack(side=tk.LEFT, padx=20, pady=10)
        
        # Status label
        status_label = tk.Label(
            header_frame,
            text="Evaluating...",
            bg="#34495e",
            fg="#f39c12",
            font=("Arial", 10)
        )
        status_label.pack(side=tk.RIGHT, padx=20)
        
        # Create treeview for results
        tree_frame = tk.Frame(eval_window, bg="#2c3e50")
        tree_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbars
        tree_scroll_y = tk.Scrollbar(tree_frame)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        tree_scroll_x = tk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Treeview
        columns = ("Type", "Parameter", "Match Ratio", "Repeatability", "Exec Time (ms)", "KP Original", "KP Augmented")
        tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set,
            height=20
        )
        
        tree_scroll_y.config(command=tree.yview)
        tree_scroll_x.config(command=tree.xview)
        
        # Define columns
        tree.heading("Type", text="Augmentation Type")
        tree.heading("Parameter", text="Parameter")
        tree.heading("Match Ratio", text="Match Ratio")
        tree.heading("Repeatability", text="Repeatability")
        tree.heading("Exec Time (ms)", text="Exec Time (ms)")
        tree.heading("KP Original", text="KP Original")
        tree.heading("KP Augmented", text="KP Augmented")
        
        tree.column("Type", width=120)
        tree.column("Parameter", width=100)
        tree.column("Match Ratio", width=100)
        tree.column("Repeatability", width=120)
        tree.column("Exec Time (ms)", width=120)
        tree.column("KP Original", width=100)
        tree.column("KP Augmented", width=120)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bottom button frame
        button_frame = tk.Frame(eval_window, bg="#34495e", height=60)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)
        button_frame.pack_propagate(False)
        
        # Export button (initially disabled)
        export_button = tk.Button(
            button_frame,
            text="📥 Export to Excel",
            command=lambda: self.export_evaluation_results(results_data, detector_choice),
            bg="#27ae60",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=5,
            state=tk.DISABLED
        )
        export_button.pack(side=tk.RIGHT, padx=20, pady=10)
        
        tk.Button(
            button_frame,
            text="Close",
            command=eval_window.destroy,
            bg="#c0392b",
            fg="white",
            font=("Arial", 10),
            padx=15,
            pady=5
        ).pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Store results for export
        results_data = []
        
        # Run evaluation in separate thread
        def run_evaluation():
            try:
                # Create detector function based on selection
                def detector_func(image):
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                    
                    if detector_choice == "orb":
                        orb = cv2.ORB_create(nfeatures=MAX_KEYPOINTS_ORB)
                        kp, desc = orb.detectAndCompute(gray, None)
                        return kp, desc
                    elif detector_choice == "sift":
                        if hasattr(cv2, 'SIFT_create'):
                            sift = cv2.SIFT_create(nfeatures=MAX_KEYPOINTS_SIFT)
                            kp, desc = sift.detectAndCompute(gray, None)
                            return kp, desc
                        else:
                            raise Exception("SIFT not available")
                    elif detector_choice == "surf":
                        if hasattr(cv2, 'xfeatures2d'):
                            surf = cv2.xfeatures2d.SURF_create(MAX_KEYPOINTS_SURF)
                            kp, desc = surf.detectAndCompute(gray, None)
                            return kp, desc
                        else:
                            raise Exception("SURF not available")
                    else:
                        # Fallback to ORB
                        orb = cv2.ORB_create(nfeatures=MAX_KEYPOINTS_ORB)
                        kp, desc = orb.detectAndCompute(gray, None)
                        return kp, desc
                
                # Create evaluator
                evaluator = AugmentationEvaluator(detector_func)
                
                # Run evaluation
                results = evaluator.evaluate_all(saved_frame.frame)
                
                # Update UI in main thread
                def update_ui():
                    for result in results:
                        tree.insert("", tk.END, values=(
                            result['type'],
                            result['param'],
                            f"{result['match_ratio']:.3f}",
                            f"{result['repeatability']:.3f}",
                            f"{result['exec_time_ms']:.2f}",
                            result['keypoints_original'],
                            result['keypoints_augmented']
                        ))
                    
                    status_label.config(text="✓ Evaluation Complete", fg="#27ae60")
                    export_button.config(state=tk.NORMAL)
                    results_data.extend(results)
                
                eval_window.after(0, update_ui)
                
            except Exception as e:
                def show_error():
                    status_label.config(text=f"✗ Error: {str(e)}", fg="#c0392b")
                    messagebox.showerror("Evaluation Error", f"An error occurred:\n{str(e)}")
                
                eval_window.after(0, show_error)
        
        # Start evaluation thread
        eval_thread = threading.Thread(target=run_evaluation, daemon=True)
        eval_thread.start()
    
    def export_evaluation_results(self, results: List[Dict], detector_name: str):
        """Export evaluation results to Excel."""
        if not results:
            messagebox.showwarning("Warning", "No results to export.")
            return
        
        # Ask for save location
        filename = filedialog.asksaveasfilename(
            title="Export Evaluation Results",
            defaultextension=".xlsx",
            filetypes=[
                ("Excel Files", "*.xlsx"),
                ("All Files", "*.*")
            ]
        )
        
        if filename:
            try:
                success = export_to_excel(results, detector_name.upper(), filename)
                if success:
                    messagebox.showinfo("Success", f"Results exported to:\n{filename}")
                else:
                    messagebox.showerror("Error", "Failed to export results.")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def on_closing(self):
        """Handle application closing."""
        self.pause_video()
        self.video_manager.release()
        self.root.destroy()


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = VideoFilterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
