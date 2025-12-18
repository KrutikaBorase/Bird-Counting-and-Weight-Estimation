"""
Video processing pipeline for bird counting and weight estimation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from .bird_detector import BirdDetector
from .bird_tracker import BirdTracker
from .weight_estimator import WeightEstimator
from .utils import (
    draw_bbox_with_label,
    draw_count_overlay,
    extract_video_metadata,
    generate_output_filename,
    save_json_result,
    sample_list
)
from .config import (
    VIDEO_OUTPUT_DIR,
    JSON_OUTPUT_DIR,
    VIDEO_CODEC,
    OUTPUT_FPS,
    BOX_COLOR,
    DEFAULT_FPS_SAMPLE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Main video processing pipeline.
    
    Orchestrates detection, tracking, and weight estimation for video analysis.
    """
    
    def __init__(
        self,
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.4,
        fps_sample: int = DEFAULT_FPS_SAMPLE
    ):
        """
        Initialize video processor.
        
        Args:
            conf_thresh: Detection confidence threshold
            iou_thresh: IoU threshold for NMS
            fps_sample: Process every Nth frame
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.fps_sample = fps_sample
        
        # Components will be initialized when processing starts
        self.detector: Optional[BirdDetector] = None
        self.tracker: Optional[BirdTracker] = None
        self.weight_estimator: Optional[WeightEstimator] = None
        
        logger.info(f"VideoProcessor initialized with fps_sample={fps_sample}")
    
    def process_video(
        self,
        video_path: Path,
        output_video_name: Optional[str] = None,
        output_json_name: Optional[str] = None
    ) -> Dict:
        """
        Process video and generate analysis results.
        
        Args:
            video_path: Path to input video
            output_video_name: Name for output video (auto-generated if None)
            output_json_name: Name for output JSON (auto-generated if None)
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Processing video: {video_path}")
        
        # Extract video metadata
        metadata = extract_video_metadata(video_path)
        logger.info(f"Video metadata: {metadata}")
        
        # Initialize components
        self.detector = BirdDetector(
            conf_thresh=self.conf_thresh,
            iou_thresh=self.iou_thresh
        )
        self.tracker = BirdTracker()
        self.weight_estimator = WeightEstimator(
            frame_width=metadata['width'],
            frame_height=metadata['height']
        )
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        # Prepare output video
        if output_video_name is None:
            output_video_name = generate_output_filename("annotated", "mp4")
        output_video_path = VIDEO_OUTPUT_DIR / output_video_name
        
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        out = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            OUTPUT_FPS,
            (metadata['width'], metadata['height'])
        )
        
        # Process frames
        frame_number = 0
        processed_frames = 0
        count_data = []
        all_weight_estimates = []
        
        logger.info("Starting frame processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            
            # Sample frames
            if frame_number % self.fps_sample != 0:
                # Still write frame to output video
                out.write(frame)
                continue
            
            processed_frames += 1
            
            # Detect birds
            bboxes, confidences, class_ids = self.detector.detect(frame)
            
            # Update tracker
            tracks = self.tracker.update(bboxes, confidences, frame)
            
            # Estimate weights
            weight_estimates = self.weight_estimator.estimate_batch(tracks)
            all_weight_estimates.extend(weight_estimates)
            
            # Get current count
            count = len(tracks)
            timestamp = frame_number / metadata['fps']
            
            count_data.append({
                'timestamp': timestamp,
                'frame_number': frame_number,
                'count': count
            })
            
            # Annotate frame
            annotated_frame = self._annotate_frame(
                frame.copy(),
                tracks,
                weight_estimates,
                count
            )
            
            # Write annotated frame
            out.write(annotated_frame)
            
            # Log progress
            if processed_frames % 30 == 0:
                logger.info(f"Processed {processed_frames} frames (frame {frame_number}/{metadata['frame_count']})")
        
        # Release resources
        cap.release()
        out.release()
        
        logger.info(f"Video processing complete. Processed {processed_frames} frames.")
        
        # Prepare results
        results = self._prepare_results(
            count_data,
            all_weight_estimates,
            output_video_path,
            metadata
        )
        
        # Save JSON
        if output_json_name is None:
            output_json_name = generate_output_filename("analysis", "json")
        output_json_path = JSON_OUTPUT_DIR / output_json_name
        save_json_result(results, output_json_path)
        
        logger.info(f"Results saved to {output_json_path}")
        
        return results
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        tracks: List[Tuple[int, np.ndarray, float]],
        weight_estimates: List[Dict],
        count: int
    ) -> np.ndarray:
        """
        Annotate frame with detections, tracks, and weights.
        
        Args:
            frame: Input frame
            tracks: List of (track_id, bbox, confidence) tuples
            weight_estimates: List of weight estimate dictionaries
            count: Current bird count
            
        Returns:
            Annotated frame
        """
        # Create weight lookup
        weight_lookup = {est['track_id']: est for est in weight_estimates}
        
        # Draw each track
        for track_id, bbox, confidence in tracks:
            # Get weight estimate
            weight_info = weight_lookup.get(track_id, {})
            weight_value = weight_info.get('weight_value', 0.0)
            
            # Create label
            label = f"ID:{track_id} W:{weight_value:.1f}"
            
            # Draw bounding box with label
            frame = draw_bbox_with_label(
                frame,
                bbox,
                label,
                color=BOX_COLOR
            )
        
        # Draw count overlay
        frame = draw_count_overlay(frame, count)
        
        return frame
    
    def _prepare_results(
        self,
        count_data: List[Dict],
        weight_estimates: List[Dict],
        output_video_path: Path,
        metadata: Dict
    ) -> Dict:
        """
        Prepare final analysis results.
        
        Args:
            count_data: List of count data points
            weight_estimates: List of weight estimates
            output_video_path: Path to output video
            metadata: Video metadata
            
        Returns:
            Complete analysis results dictionary
        """
        # Get tracking statistics
        track_stats = self.tracker.get_track_statistics()
        
        # Get weight statistics
        weight_stats = self.weight_estimator.get_aggregate_statistics()
        
        # Sample tracks for response
        all_track_ids = self.tracker.get_all_track_ids()
        sampled_track_ids = sample_list(all_track_ids, max_samples=10)
        
        tracks_sample = []
        for track_id in sampled_track_ids:
            history = self.tracker.get_track_history(track_id)
            if len(history) > 0:
                # Get a representative frame (middle of track)
                mid_idx = len(history) // 2
                track_info = history[mid_idx]
                
                tracks_sample.append({
                    'track_id': track_id,
                    'bbox': {
                        'x1': track_info['bbox'][0],
                        'y1': track_info['bbox'][1],
                        'x2': track_info['bbox'][2],
                        'y2': track_info['bbox'][3]
                    },
                    'confidence': track_info['confidence'],
                    'frame_number': track_info['frame']
                })
        
        # Sample weight estimates
        sampled_weights = sample_list(weight_estimates, max_samples=20)
        
        # Calculate average weight per track
        avg_weights_per_track = []
        for track_id in all_track_ids:
            avg_weight = self.weight_estimator.get_average_weight(track_id)
            if avg_weight is not None:
                avg_weights_per_track.append({
                    'track_id': track_id,
                    'weight_value': avg_weight,
                    'unit': 'index',
                    'confidence': 0.7,  # Average confidence
                    'uncertainty': self.weight_estimator._calculate_uncertainty(track_id),
                    'method': 'bbox_area_with_perspective_averaged'
                })
        
        # Prepare final results
        results = {
            'counts': count_data,
            'tracks_sample': tracks_sample,
            'weight_estimates': avg_weights_per_track,
            'artifacts': {
                'output_video': str(output_video_path),
                'output_json': str(JSON_OUTPUT_DIR / generate_output_filename("analysis", "json"))
            },
            'metadata': {
                'input_video': {
                    'fps': metadata['fps'],
                    'frame_count': metadata['frame_count'],
                    'width': metadata['width'],
                    'height': metadata['height'],
                    'duration_seconds': metadata['duration_seconds']
                },
                'processing': {
                    'fps_sample': self.fps_sample,
                    'conf_thresh': self.conf_thresh,
                    'iou_thresh': self.iou_thresh
                },
                'tracking_statistics': track_stats,
                'weight_statistics': weight_stats
            },
            'calibration_notes': self.weight_estimator.get_calibration_notes()
        }
        
        return results
