import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from scipy import ndimage


class ImageAugmenter:
  
    @staticmethod
    def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, degrees, 1.0)
        
        cos_val = np.abs(M[0, 0])
        sin_val = np.abs(M[0, 1])
        new_w = int((h * sin_val) + (w * cos_val))
        new_h = int((h * cos_val) + (w * sin_val))
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(0, 0, 0))
        return rotated
    
    @staticmethod
    def scale(image: np.ndarray, factor: float) -> np.ndarray:
        h, w = image.shape[:2]
        new_w = int(w * factor)
        new_h = int(h * factor)
        
        if factor < 1:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC
            
        return cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    @staticmethod
    def blur(image: np.ndarray, sigma: float) -> np.ndarray:
        ksize = int(2 * np.ceil(3 * sigma) + 1)
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)
    
    @staticmethod
    def mirror(image: np.ndarray, horizontal: bool = False, 
               vertical: bool = False) -> np.ndarray:
        if horizontal and vertical:
            return cv2.flip(image, -1)
        elif horizontal:
            return cv2.flip(image, 1)
        elif vertical:
            return cv2.flip(image, 0)
        else:
            return image.copy()
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, percentage: float) -> np.ndarray:
        factor = 1.0 + (percentage / 100.0)
        adjusted = np.clip(image * factor, 0, 255).astype(np.uint8)
        return adjusted


class AugmentationEvaluator:  
    def __init__(self, detector_func: callable):
        self.detector_func = detector_func
    
    def measure_execution_time(self, image: np.ndarray) -> float:
        start = time.time()
        try:
            self.detector_func(image)
        except:
            pass
        end = time.time()
        return (end - start) * 1000  # Convert to ms
    
    @staticmethod
    def hamming_distance(desc1: np.ndarray, desc2: np.ndarray) -> int:
        if desc1.dtype == np.uint8 and desc2.dtype == np.uint8:
            return np.count_nonzero(desc1 != desc2)
        else:
            return int(np.sum(desc1 != desc2))
    
    @staticmethod
    def euclidean_distance(desc1: np.ndarray, desc2: np.ndarray) -> float:
        """
        Calculate Euclidean distance for float descriptors (SIFT, SURF).
        
        Args:
            desc1: First descriptor
            desc2: Second descriptor
            
        Returns:
            Euclidean distance
        """
        return np.linalg.norm(desc1 - desc2)
    
    def match_descriptors(self, desc1: List[np.ndarray], 
                         desc2: List[np.ndarray],
                         threshold: Optional[float] = None) -> float:
        """
        Match descriptors between two sets using Lowe's ratio test.
        
        Args:
            desc1: First set of descriptors
            desc2: Second set of descriptors
            threshold: Distance threshold for matching
            
        Returns:
            Match ratio (0.0 to 1.0)
        """
        if not desc1 or not desc2 or len(desc1) == 0 or len(desc2) == 0:
            return 0.0
        
        is_binary = desc1[0].dtype == np.uint8
        
        if not is_binary:
            desc1_normalized = []
            for d in desc1:
                norm = np.linalg.norm(d)
                if norm > 0:
                    desc1_normalized.append(d / norm)
                else:
                    desc1_normalized.append(d)
            
            desc2_normalized = []
            for d in desc2:
                norm = np.linalg.norm(d)
                if norm > 0:
                    desc2_normalized.append(d / norm)
                else:
                    desc2_normalized.append(d)
            
            desc1 = desc1_normalized
            desc2 = desc2_normalized
        
        if threshold is None:
            if is_binary:
                threshold = 60
            else:
                threshold = 0.8
        
        matched = 0
        for d1 in desc1:
            best_dist = float('inf')
            second_dist = float('inf')
            
            for d2 in desc2:
                if is_binary:
                    dist = self.hamming_distance(d1, d2)
                else:
                    dist = self.euclidean_distance(d1, d2)
                
                if dist < best_dist:
                    second_dist = best_dist
                    best_dist = dist
                elif dist < second_dist:
                    second_dist = dist
    
            if second_dist > 0:
                ratio = best_dist / second_dist
            else:
                ratio = 1.0
            
            if best_dist < threshold and ratio < 0.8:
                matched += 1
        
        return matched / len(desc1) if len(desc1) > 0 else 0.0
    
    def compute_repeatability(self, kp1: List[cv2.KeyPoint],
                            kp2: List[cv2.KeyPoint],
                            transform_info: Dict,
                            epsilon: float = 5.0) -> float:
        """
        Compute keypoint repeatability after transformation.
        
        Args:
            kp1: Original keypoints
            kp2: Transformed keypoints
            transform_info: Information about the transformation
            epsilon: Distance threshold for matching (pixels)
            
        Returns:
            Repeatability score (0.0 to 1.0)
        """
        if not kp1 or not kp2 or len(kp1) == 0 or len(kp2) == 0:
            return 0.0
        
        # Transform keypoints from image 1 to image 2 coordinate space
        transformed_kp1 = self.transform_keypoints(kp1, transform_info)
        
        matched = 0
        img2_width = transform_info.get('width', float('inf'))
        img2_height = transform_info.get('height', float('inf'))
        
        for pt1 in transformed_kp1:
            # Check if transformed point is within bounds
            if pt1[0] < 0 or pt1[1] < 0 or pt1[0] >= img2_width or pt1[1] >= img2_height:
                continue
            
            # Find nearest neighbor in kp2
            for kp in kp2:
                dx = kp.pt[0] - pt1[0]
                dy = kp.pt[1] - pt1[1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist <= epsilon:
                    matched += 1
                    break
        
        return matched / len(transformed_kp1) if len(transformed_kp1) > 0 else 0.0
    
    def transform_keypoints(self, keypoints: List[cv2.KeyPoint],
                           transform_info: Dict) -> List[Tuple[float, float]]:
        """
        Transform keypoints according to augmentation using proper transformation matrices.
        
        Args:
            keypoints: Original keypoints
            transform_info: Transformation parameters
            
        Returns:
            List of transformed (x, y) coordinates
        """
        import cv2
        
        trans_type = transform_info.get('type', 'none')
        param = transform_info.get('param', 0)
        
        transformed = []
        
        # Extract keypoint coordinates
        points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
        
        if trans_type == 'rotate':
            # Use OpenCV's rotation matrix (same as used in augmentation)
            src_w = transform_info.get('src_width', 0)
            src_h = transform_info.get('src_height', 0)
            center = (src_w / 2, src_h / 2)
            
            # Get rotation matrix (same as ImageAugmenter.rotate uses)
            M = cv2.getRotationMatrix2D(center, param, 1.0)
            
            # Calculate new dimensions (same as augmentation)
            cos_val = np.abs(M[0, 0])
            sin_val = np.abs(M[0, 1])
            new_w = int((src_h * sin_val) + (src_w * cos_val))
            new_h = int((src_h * cos_val) + (src_w * sin_val))
            
            # Adjust translation (same as augmentation)
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            # Transform points using the matrix
            # Add ones for homogeneous coordinates
            ones = np.ones((points.shape[0], 1), dtype=np.float32)
            points_homogeneous = np.hstack([points, ones])
            
            # Apply transformation: M @ [x, y, 1]^T
            transformed_points = (M @ points_homogeneous.T).T
            
            for pt in transformed_points:
                transformed.append((float(pt[0]), float(pt[1])))
        
        elif trans_type == 'scale':
            # Scaling transformation
            scale_factor = param
            for pt in points:
                transformed.append((float(pt[0] * scale_factor), float(pt[1] * scale_factor)))
        
        elif trans_type == 'mirror':
            # Mirror transformation
            src_w = transform_info.get('src_width', transform_info.get('width', 0))
            src_h = transform_info.get('src_height', transform_info.get('height', 0))
            
            if param == (True, False):  # Horizontal flip
                for pt in points:
                    transformed.append((float(src_w - pt[0]), float(pt[1])))
            elif param == (False, True):  # Vertical flip
                for pt in points:
                    transformed.append((float(pt[0]), float(src_h - pt[1])))
            elif param == (True, True):  # Both
                for pt in points:
                    transformed.append((float(src_w - pt[0]), float(src_h - pt[1])))
            else:
                for pt in points:
                    transformed.append((float(pt[0]), float(pt[1])))
        
        else:
            # No transformation or blur/brightness (keypoints stay in place)
            for pt in points:
                transformed.append((float(pt[0]), float(pt[1])))
        
        return transformed
    
    def evaluate_augmentation(self, original_image: np.ndarray,
                            augmentation_type: str,
                            augmentation_params: List) -> List[Dict]:
        """
        Evaluate detector on a specific augmentation type.
        
        Args:
            original_image: Original image
            augmentation_type: Type of augmentation
            augmentation_params: List of parameters to test
            
        Returns:
            List of evaluation results
        """
        augmenter = ImageAugmenter()
        results = []
        
        # Detect features in original
        try:
            base_kp, base_desc = self.detector_func(original_image)
        except Exception as e:
            print(f"Error detecting base features: {e}")
            return []
        
        if base_desc is None or len(base_desc) == 0:
            print("No base descriptors found")
            return []
        
        # Convert descriptors to list format
        base_desc_list = [base_desc[i] for i in range(len(base_desc))]
        
        for param in augmentation_params:
            try:
                # Apply augmentation
                if augmentation_type == 'rotate':
                    augmented = augmenter.rotate(original_image, param)
                    param_str = f"{param}°"
                elif augmentation_type == 'scale':
                    augmented = augmenter.scale(original_image, param)
                    param_str = f"{param}×"
                elif augmentation_type == 'blur':
                    augmented = augmenter.blur(original_image, param)
                    param_str = f"σ={param}"
                elif augmentation_type == 'mirror':
                    h, v = param
                    augmented = augmenter.mirror(original_image, h, v)
                    param_str = "H" if h else "" + "V" if v else ""
                    if not param_str:
                        param_str = "none"
                elif augmentation_type == 'brightness':
                    augmented = augmenter.adjust_brightness(original_image, param)
                    param_str = f"{'+' if param > 0 else ''}{param}%"
                else:
                    continue
                
                # Detect features in augmented image
                aug_kp, aug_desc = self.detector_func(augmented)
                
                if aug_desc is None or len(aug_desc) == 0:
                    results.append({
                        'type': augmentation_type,
                        'param': param_str,
                        'match_ratio': 0.0,
                        'repeatability': 0.0,
                        'exec_time_ms': 0.0,
                        'keypoints_original': len(base_kp) if base_kp else 0,
                        'keypoints_augmented': 0
                    })
                    continue
                
                # Convert augmented descriptors to list
                aug_desc_list = [aug_desc[i] for i in range(len(aug_desc))]
                
                # Calculate metrics
                match_ratio = self.match_descriptors(base_desc_list, aug_desc_list)
                
                transform_info = {
                    'type': augmentation_type,
                    'param': param,
                    'width': augmented.shape[1],
                    'height': augmented.shape[0],
                    'src_width': original_image.shape[1],
                    'src_height': original_image.shape[0]
                }
                
                repeatability = self.compute_repeatability(base_kp, aug_kp, transform_info)
                exec_time = self.measure_execution_time(augmented)
                
                results.append({
                    'type': augmentation_type,
                    'param': param_str,
                    'match_ratio': match_ratio,
                    'repeatability': repeatability,
                    'exec_time_ms': exec_time,
                    'keypoints_original': len(base_kp) if base_kp else 0,
                    'keypoints_augmented': len(aug_kp) if aug_kp else 0
                })
                
            except Exception as e:
                print(f"Error processing {augmentation_type} with param {param}: {e}")
                results.append({
                    'type': augmentation_type,
                    'param': str(param),
                    'match_ratio': 0.0,
                    'repeatability': 0.0,
                    'exec_time_ms': 0.0,
                    'keypoints_original': len(base_kp) if base_kp else 0,
                    'keypoints_augmented': 0
                })
        
        return results
    
    def evaluate_all(self, image: np.ndarray) -> List[Dict]:
        """
        Evaluate detector against all standard augmentations.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of evaluation results
        """
        all_results = []
        
        print("Evaluating rotations...")
        all_results.extend(self.evaluate_augmentation(
            image, 'rotate', [15, 30, 45, 60, 90]
        ))
        
        print("Evaluating scaling...")
        all_results.extend(self.evaluate_augmentation(
            image, 'scale', [0.5, 0.75, 1.25, 1.5, 2.0]
        ))
        
        print("Evaluating blur...")
        all_results.extend(self.evaluate_augmentation(
            image, 'blur', [0.5, 1.0, 1.5, 2.0, 3.0]
        ))
        
        print("Evaluating mirroring...")
        all_results.extend(self.evaluate_augmentation(
            image, 'mirror', [
                (True, False),   # Horizontal
                (False, True),   # Vertical
                (True, True)     # Both
            ]
        ))
        
        print("Evaluating brightness...")
        all_results.extend(self.evaluate_augmentation(
            image, 'brightness', [-50, -25, 25, 50, 75]
        ))
        
        return all_results


def export_to_excel(results: List[Dict], detector_name: str, filename: str):
    """
    Export evaluation results to Excel file.
    
    Args:
        results: Evaluation results
        detector_name: Name of the detector
        filename: Output filename
    """
    try:
        df = pd.DataFrame(results)
        
        # Format columns
        df['match_ratio'] = df['match_ratio'].apply(lambda x: f"{x:.3f}")
        df['repeatability'] = df['repeatability'].apply(lambda x: f"{x:.3f}")
        df['exec_time_ms'] = df['exec_time_ms'].apply(lambda x: f"{x:.2f}")
        
        # Rename columns for better readability
        df = df.rename(columns={
            'type': 'Augmentation Type',
            'param': 'Parameter',
            'match_ratio': 'Match Ratio',
            'repeatability': 'Repeatability',
            'exec_time_ms': 'Execution Time (ms)',
            'keypoints_original': 'Keypoints (Original)',
            'keypoints_augmented': 'Keypoints (Augmented)'
        })
        
        # Create Excel writer
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=detector_name, index=False)
            
            # Get workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets[detector_name]
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        return True
    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        return False

