"""
Utility functions for video processing and visualization.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import json
from datetime import datetime


def draw_bbox_with_label(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.6
) -> np.ndarray:
    """
    Draw bounding box with label on frame.
    
    Args:
        frame: Input frame
        bbox: Bounding box (x1, y1, x2, y2)
        label: Text label to display
        color: Box color in BGR
        thickness: Line thickness
        font_scale: Font scale for text
        
    Returns:
        Frame with drawn bounding box and label
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Calculate text size for background
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Draw text background
    cv2.rectangle(
        frame,
        (x1, y1 - text_height - baseline - 5),
        (x1 + text_width, y1),
        color,
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        label,
        (x1, y1 - baseline - 2),
        font,
        font_scale,
        (0, 0, 0),  # Black text
        thickness
    )
    
    return frame


def draw_count_overlay(
    frame: np.ndarray,
    count: int,
    position: Tuple[int, int] = (20, 50),
    font_scale: float = 1.2,
    thickness: int = 3
) -> np.ndarray:
    """
    Draw bird count overlay on frame.
    
    Args:
        frame: Input frame
        count: Bird count to display
        position: Text position (x, y)
        font_scale: Font scale
        thickness: Text thickness
        
    Returns:
        Frame with count overlay
    """
    text = f"Bird Count: {count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Draw text with black outline for visibility
    cv2.putText(frame, text, position, font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, position, font, font_scale, (255, 255, 255), thickness)
    
    return frame


def calculate_bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    """
    Calculate bounding box area.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        Area in pixels²
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def calculate_bbox_aspect_ratio(bbox: Tuple[float, float, float, float]) -> float:
    """
    Calculate bounding box aspect ratio.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        Aspect ratio (width / height)
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return width / height if height > 0 else 0


def get_bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Get center point of bounding box.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        Center coordinates (cx, cy)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def save_json_result(result: Dict, output_path: Path) -> None:
    """
    Save analysis result to JSON file.
    
    Args:
        result: Result dictionary
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)


def generate_output_filename(prefix: str, extension: str) -> str:
    """
    Generate timestamped output filename.
    
    Args:
        prefix: Filename prefix
        extension: File extension (without dot)
        
    Returns:
        Timestamped filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def extract_video_metadata(video_path: Path) -> Dict:
    """
    Extract metadata from video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video metadata
    """
    cap = cv2.VideoCapture(str(video_path))
    
    metadata = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_seconds": 0
    }
    
    if metadata["fps"] > 0:
        metadata["duration_seconds"] = metadata["frame_count"] / metadata["fps"]
    
    cap.release()
    return metadata


def sample_list(items: List, max_samples: int = 10) -> List:
    """
    Sample items from list evenly.
    
    Args:
        items: List of items
        max_samples: Maximum number of samples
        
    Returns:
        Sampled list
    """
    if len(items) <= max_samples:
        return items
    
    step = len(items) / max_samples
    indices = [int(i * step) for i in range(max_samples)]
    return [items[i] for i in indices]
