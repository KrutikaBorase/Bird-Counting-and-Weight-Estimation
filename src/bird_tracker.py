"""
Bird tracking using DeepSORT algorithm.
"""

from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import logging

from .config import MAX_AGE, MIN_HITS, IOU_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BirdTracker:
    """
    Bird tracking using DeepSORT algorithm.
    
    Maintains stable tracking IDs across frames, handles occlusions,
    and minimizes ID switches.
    """
    
    def __init__(
        self,
        max_age: int = MAX_AGE,
        min_hits: int = MIN_HITS,
        iou_threshold: float = IOU_THRESHOLD
    ):
        """
        Initialize bird tracker.
        
        Args:
            max_age: Maximum frames to keep alive a track without detections
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IoU threshold for track matching
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=min_hits,
            nms_max_overlap=iou_threshold,
            max_cosine_distance=0.3,
            nn_budget=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=False  # Set to True if GPU available
        )
        
        # Track history for analysis
        self.track_history: Dict[int, List[Dict]] = defaultdict(list)
        self.frame_counts: List[int] = []
        self.current_frame = 0
        
        logger.info(f"Tracker initialized with max_age={max_age}, min_hits={min_hits}")
    
    def update(
        self,
        bboxes: List[np.ndarray],
        confidences: List[float],
        frame: np.ndarray
    ) -> List[Tuple[int, np.ndarray, float]]:
        """
        Update tracker with new detections.
        
        Args:
            bboxes: List of bounding boxes [x1, y1, x2, y2]
            confidences: List of confidence scores
            frame: Current frame (for feature extraction)
            
        Returns:
            List of (track_id, bbox, confidence) tuples
        """
        self.current_frame += 1
        
        # Prepare detections in format required by DeepSORT
        # Format: ([x1, y1, w, h], confidence, detection_class)
        detections = []
        for bbox, conf in zip(bboxes, confidences):
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], conf, 'bird'))
        
        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Extract confirmed tracks
        active_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()  # Get bbox in [x1, y1, x2, y2] format
            confidence = track.get_det_conf()
            
            if confidence is None:
                confidence = 0.0
            
            active_tracks.append((track_id, ltrb, confidence))
            
            # Store track history
            self.track_history[track_id].append({
                'frame': self.current_frame,
                'bbox': ltrb.tolist(),
                'confidence': float(confidence)
            })
        
        # Store count for this frame
        count = len(active_tracks)
        self.frame_counts.append(count)
        
        return active_tracks
    
    def get_count(self) -> int:
        """
        Get current bird count.
        
        Returns:
            Number of active tracks
        """
        if len(self.frame_counts) > 0:
            return self.frame_counts[-1]
        return 0
    
    def get_count_history(self) -> List[int]:
        """
        Get count history over all frames.
        
        Returns:
            List of counts per frame
        """
        return self.frame_counts.copy()
    
    def get_track_history(self, track_id: int) -> List[Dict]:
        """
        Get history for a specific track.
        
        Args:
            track_id: Track ID
            
        Returns:
            List of track states over time
        """
        return self.track_history.get(track_id, [])
    
    def get_all_track_ids(self) -> List[int]:
        """
        Get all track IDs that have been seen.
        
        Returns:
            List of track IDs
        """
        return list(self.track_history.keys())
    
    def get_track_statistics(self) -> Dict:
        """
        Get tracking statistics.
        
        Returns:
            Dictionary with tracking statistics
        """
        total_tracks = len(self.track_history)
        avg_count = np.mean(self.frame_counts) if self.frame_counts else 0
        max_count = max(self.frame_counts) if self.frame_counts else 0
        min_count = min(self.frame_counts) if self.frame_counts else 0
        
        # Calculate track lifespans
        track_lifespans = [len(history) for history in self.track_history.values()]
        avg_lifespan = np.mean(track_lifespans) if track_lifespans else 0
        
        return {
            'total_unique_tracks': total_tracks,
            'average_count': float(avg_count),
            'max_count': int(max_count),
            'min_count': int(min_count),
            'average_track_lifespan': float(avg_lifespan),
            'total_frames_processed': self.current_frame
        }
    
    def reset(self):
        """Reset tracker state."""
        self.tracker = DeepSort(
            max_age=MAX_AGE,
            n_init=MIN_HITS,
            nms_max_overlap=IOU_THRESHOLD,
            max_cosine_distance=0.3,
            nn_budget=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=False
        )
        self.track_history.clear()
        self.frame_counts.clear()
        self.current_frame = 0
        logger.info("Tracker reset")
