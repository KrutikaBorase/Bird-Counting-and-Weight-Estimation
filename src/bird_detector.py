"""
Bird detection using YOLOv8.
"""

from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import logging

from .config import (
    YOLO_MODEL_PATH,
    YOLO_MODEL,
    DEFAULT_CONF_THRESH,
    DEFAULT_IOU_THRESH
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BirdDetector:
    """
    Bird detection using YOLOv8 model.
    
    This class handles loading the YOLO model and performing inference
    on video frames to detect birds.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        conf_thresh: float = DEFAULT_CONF_THRESH,
        iou_thresh: float = DEFAULT_IOU_THRESH
    ):
        """
        Initialize bird detector.
        
        Args:
            model_path: Path to YOLO model weights (if None, downloads pretrained)
            conf_thresh: Confidence threshold for detections
            iou_thresh: IoU threshold for NMS
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        # Load YOLO model
        if model_path is None or not Path(model_path).exists():
            logger.info(f"Loading pretrained YOLOv8 model: {YOLO_MODEL}")
            self.model = YOLO(YOLO_MODEL)
        else:
            logger.info(f"Loading YOLO model from: {model_path}")
            self.model = YOLO(str(model_path))
        
        logger.info(f"Detector initialized with conf={conf_thresh}, iou={iou_thresh}")
    
    def detect(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """
        Detect birds in a single frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (bboxes, confidences, class_ids)
            - bboxes: List of bounding boxes [x1, y1, x2, y2]
            - confidences: List of confidence scores
            - class_ids: List of class IDs
        """
        # Run inference
        results = self.model(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False
        )
        
        bboxes = []
        confidences = []
        class_ids = []
        
        # Extract detections
        if len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                # Filter for bird class (class 14 in COCO dataset)
                # Note: If using a custom-trained model, adjust this filter
                for box, conf, cls in zip(boxes, confs, classes):
                    # For pretrained COCO model, class 14 is 'bird'
                    # If all detections should be considered as birds, remove this filter
                    # or train a custom model on poultry dataset
                    if cls == 14:  # Bird class in COCO
                        bboxes.append(box)
                        confidences.append(float(conf))
                        class_ids.append(int(cls))
        
        return bboxes, confidences, class_ids
    
    def detect_all_objects(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """
        Detect all objects in frame (useful when using custom-trained model).
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (bboxes, confidences, class_ids)
        """
        results = self.model(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False
        )
        
        bboxes = []
        confidences = []
        class_ids = []
        
        if len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls in zip(boxes, confs, classes):
                    bboxes.append(box)
                    confidences.append(float(conf))
                    class_ids.append(int(cls))
        
        return bboxes, confidences, class_ids
    
    def update_thresholds(self, conf_thresh: Optional[float] = None, iou_thresh: Optional[float] = None):
        """
        Update detection thresholds.
        
        Args:
            conf_thresh: New confidence threshold
            iou_thresh: New IoU threshold
        """
        if conf_thresh is not None:
            self.conf_thresh = conf_thresh
            logger.info(f"Updated confidence threshold to {conf_thresh}")
        
        if iou_thresh is not None:
            self.iou_thresh = iou_thresh
            logger.info(f"Updated IoU threshold to {iou_thresh}")
