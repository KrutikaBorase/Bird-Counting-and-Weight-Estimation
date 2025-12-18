"""
Weight estimation using bounding box features and proxy calculation.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

from .config import WEIGHT_PROXY_SCALE, MIN_BOX_AREA
from .utils import calculate_bbox_area, calculate_bbox_aspect_ratio, get_bbox_center

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightEstimator:
    """
    Weight estimation using visual features from bounding boxes.
    
    Since ground truth weights are typically unavailable in video datasets,
    this class calculates a weight proxy/index based on bounding box features.
    
    The proxy can be calibrated to actual weights using:
    - Reference objects of known size in the frame
    - Manual annotations correlating box sizes to actual weights
    - Calibration videos with known bird weights
    """
    
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        calibration_factor: float = WEIGHT_PROXY_SCALE,
        use_perspective_correction: bool = True
    ):
        """
        Initialize weight estimator.
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            calibration_factor: Scaling factor for weight proxy
            use_perspective_correction: Apply perspective correction based on position
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.calibration_factor = calibration_factor
        self.use_perspective_correction = use_perspective_correction
        
        # Store weight estimates for statistics
        self.weight_history: Dict[int, List[float]] = {}
        
        logger.info(f"Weight estimator initialized for {frame_width}x{frame_height} frames")
    
    def estimate_weight_proxy(
        self,
        bbox: np.ndarray,
        track_id: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Estimate weight proxy for a single bird.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            track_id: Optional track ID for history tracking
            
        Returns:
            Tuple of (weight_proxy, confidence)
        """
        # Calculate basic features
        area = calculate_bbox_area(bbox)
        aspect_ratio = calculate_bbox_aspect_ratio(bbox)
        cx, cy = get_bbox_center(bbox)
        
        # Skip very small boxes (likely false positives)
        if area < MIN_BOX_AREA:
            return 0.0, 0.0
        
        # Perspective correction factor
        # Birds closer to camera (lower in frame) appear larger
        perspective_factor = 1.0
        if self.use_perspective_correction:
            # Normalize y position (0 at top, 1 at bottom)
            y_norm = cy / self.frame_height
            # Apply correction: birds at bottom are closer, so reduce their apparent size
            # This is a simplified model; real perspective correction requires camera calibration
            perspective_factor = 1.0 - (0.3 * y_norm)
        
        # Calculate weight proxy
        # Base proxy on area, adjusted for perspective
        corrected_area = area * perspective_factor
        
        # Normalize by frame size
        normalized_area = corrected_area / (self.frame_width * self.frame_height)
        
        # Apply calibration factor
        weight_proxy = normalized_area * self.calibration_factor * 10000  # Scale to reasonable range
        
        # Calculate confidence based on bbox quality
        # Higher confidence for:
        # - Moderate aspect ratios (birds are roughly square when viewed from above)
        # - Sufficient size
        aspect_confidence = 1.0 - min(abs(aspect_ratio - 1.0), 1.0)  # Ideal ratio is 1.0
        size_confidence = min(area / (self.frame_width * self.frame_height * 0.1), 1.0)
        confidence = (aspect_confidence + size_confidence) / 2.0
        
        # Store in history
        if track_id is not None:
            if track_id not in self.weight_history:
                self.weight_history[track_id] = []
            self.weight_history[track_id].append(weight_proxy)
        
        return float(weight_proxy), float(confidence)
    
    def estimate_batch(
        self,
        tracks: List[Tuple[int, np.ndarray, float]]
    ) -> List[Dict]:
        """
        Estimate weights for multiple tracks.
        
        Args:
            tracks: List of (track_id, bbox, confidence) tuples
            
        Returns:
            List of weight estimate dictionaries
        """
        estimates = []
        
        for track_id, bbox, det_conf in tracks:
            weight_proxy, weight_conf = self.estimate_weight_proxy(bbox, track_id)
            
            estimates.append({
                'track_id': track_id,
                'weight_value': weight_proxy,
                'unit': 'index',
                'confidence': weight_conf,
                'uncertainty': self._calculate_uncertainty(track_id),
                'method': 'bbox_area_with_perspective'
            })
        
        return estimates
    
    def get_average_weight(self, track_id: int) -> Optional[float]:
        """
        Get average weight proxy for a track over its lifetime.
        
        Args:
            track_id: Track ID
            
        Returns:
            Average weight proxy or None if track not found
        """
        if track_id in self.weight_history and len(self.weight_history[track_id]) > 0:
            return float(np.mean(self.weight_history[track_id]))
        return None
    
    def get_aggregate_statistics(self) -> Dict:
        """
        Get aggregate weight statistics across all tracks.
        
        Returns:
            Dictionary with weight statistics
        """
        all_weights = []
        for weights in self.weight_history.values():
            all_weights.extend(weights)
        
        if len(all_weights) == 0:
            return {
                'mean_weight_proxy': 0.0,
                'std_weight_proxy': 0.0,
                'min_weight_proxy': 0.0,
                'max_weight_proxy': 0.0,
                'total_samples': 0
            }
        
        return {
            'mean_weight_proxy': float(np.mean(all_weights)),
            'std_weight_proxy': float(np.std(all_weights)),
            'min_weight_proxy': float(np.min(all_weights)),
            'max_weight_proxy': float(np.max(all_weights)),
            'total_samples': len(all_weights)
        }
    
    def _calculate_uncertainty(self, track_id: int) -> Optional[float]:
        """
        Calculate uncertainty for weight estimate.
        
        Args:
            track_id: Track ID
            
        Returns:
            Uncertainty value (standard deviation) or None
        """
        if track_id in self.weight_history and len(self.weight_history[track_id]) > 1:
            return float(np.std(self.weight_history[track_id]))
        return None
    
    def get_calibration_notes(self) -> str:
        """
        Get notes on calibration requirements.
        
        Returns:
            Calibration notes string
        """
        return """
Weight Proxy Calibration Notes:
================================

Current Method: Bounding box area with perspective correction

To convert weight proxy to actual grams, you need:

1. Reference Object Calibration:
   - Place objects of known size (e.g., 10cm x 10cm marker) in the frame
   - Measure their bounding box size in pixels
   - Calculate pixels-to-cm conversion factor
   - Use bird size in cm² to estimate weight based on species growth curves

2. Ground Truth Annotation:
   - Manually weigh a sample of birds
   - Record their weights and corresponding video timestamps
   - Extract bounding box features for these birds
   - Train a regression model: weight_proxy → actual_weight_grams

3. Camera Calibration:
   - Perform camera calibration to get intrinsic/extrinsic parameters
   - Apply proper perspective correction
   - Account for camera height and angle
   - This improves accuracy significantly

4. Species-Specific Models:
   - Different poultry species have different size-to-weight ratios
   - Collect species-specific training data
   - Train separate models or use species as a feature

Current proxy values are relative indices. Higher values indicate larger/heavier birds.
Typical range: 0-100 (arbitrary units, depends on camera setup and bird size).
"""
