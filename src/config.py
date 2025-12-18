"""
Configuration settings for the bird counting and weight estimation system.
"""

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Directory paths
DATA_DIR = ROOT_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
SAMPLE_DIR = DATA_DIR / "sample"
OUTPUT_DIR = ROOT_DIR / "outputs"
VIDEO_OUTPUT_DIR = OUTPUT_DIR / "videos"
JSON_OUTPUT_DIR = OUTPUT_DIR / "json"
MODELS_DIR = ROOT_DIR / "models"

# Create directories if they don't exist
for directory in [INPUT_DIR, SAMPLE_DIR, VIDEO_OUTPUT_DIR, JSON_OUTPUT_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model configuration
YOLO_MODEL = "yolov8n.pt"  # Using nano model for speed, can upgrade to yolov8s/m/l/x
YOLO_MODEL_PATH = MODELS_DIR / YOLO_MODEL

# Detection parameters (defaults, can be overridden via API)
DEFAULT_CONF_THRESH = 0.5  # Confidence threshold for detections
DEFAULT_IOU_THRESH = 0.4   # IoU threshold for NMS
DEFAULT_FPS_SAMPLE = 2     # Process every Nth frame (1 = all frames, 2 = every other frame)

# Tracking parameters
MAX_AGE = 30              # Maximum frames to keep alive a track without detections
MIN_HITS = 3              # Minimum hits before track is confirmed
IOU_THRESHOLD = 0.3       # IoU threshold for track matching

# Weight estimation parameters
# These are calibration factors - adjust based on camera setup and bird species
WEIGHT_PROXY_SCALE = 1.0  # Scaling factor for weight proxy
MIN_BOX_AREA = 100        # Minimum bounding box area to consider (pixels²)

# Video processing
VIDEO_CODEC = "mp4v"      # Codec for output videos
OUTPUT_FPS = 30           # FPS for output videos

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB max video size

# Visualization settings
BOX_COLOR = (0, 255, 0)   # Green for bounding boxes (BGR)
TEXT_COLOR = (255, 255, 255)  # White for text
BOX_THICKNESS = 2
FONT_SCALE = 0.6
FONT_THICKNESS = 2
