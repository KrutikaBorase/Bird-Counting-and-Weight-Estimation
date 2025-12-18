# Bird Counting and Weight Estimation System

**Author**: Krutika Borase  
**Project**: ML/AI Engineer Internship Assessment - Kuppismart Solutions (Livestockify)  
**Date**: December 2024

## Overview

This project implements a computer vision system for analyzing poultry farm CCTV footage to count birds and estimate weights. The system uses YOLOv8 for object detection, DeepSORT for multi-object tracking, and a feature-based approach for weight proxy calculation.

## Features

- **Bird Detection**: YOLOv8-based object detection with configurable confidence thresholds
- **Stable Tracking**: DeepSORT algorithm for consistent ID assignment across frames
- **Weight Estimation**: Bounding box feature-based weight proxy with perspective correction
- **Video Annotation**: Generates annotated videos with bounding boxes, tracking IDs, and counts
- **REST API**: FastAPI service for easy integration
- **Comprehensive Output**: JSON results with time-series counts, track samples, and weight estimates

## Project Structure

```
Task/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── bird_detector.py   # YOLOv8 detection
│   ├── bird_tracker.py    # DeepSORT tracking
│   ├── weight_estimator.py # Weight proxy calculation
│   ├── video_processor.py # Main processing pipeline
│   ├── models.py          # Pydantic data models
│   ├── config.py          # Configuration settings
│   └── utils.py           # Utility functions
├── data/
│   ├── input/             # Input videos (temporary)
│   └── sample/            # Sample dataset
├── outputs/
│   ├── videos/            # Annotated output videos
│   └── json/              # JSON analysis results
└── models/                # YOLO model weights (auto-downloaded)
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

### Installation

1. **Clone or extract the project**:
   ```bash
   cd Task
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download sample dataset**:
   - Find a poultry farm CCTV video dataset on Kaggle or similar platforms
   - Search for: "poultry farm video", "chicken detection dataset", "bird tracking dataset"
   - Place sample videos in `data/sample/` directory

## Running the API

### Start the Server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "OK",
  "version": "1.0.0"
}
```

### Analyze Video

**Basic usage**:
```bash
curl -X POST "http://localhost:8000/analyze_video" \
  -F "video=@data/sample/poultry_video.mp4"
```

**With custom parameters**:
```bash
curl -X POST "http://localhost:8000/analyze_video" \
  -F "video=@data/sample/poultry_video.mp4" \
  -F "fps_sample=2" \
  -F "conf_thresh=0.5" \
  -F "iou_thresh=0.4"
```

**Parameters**:
- `video` (required): Video file to analyze (mp4, avi, mov, mkv)
- `fps_sample` (optional, default=2): Process every Nth frame (1=all frames, 2=every other frame)
- `conf_thresh` (optional, default=0.5): Detection confidence threshold (0.0-1.0)
- `iou_thresh` (optional, default=0.4): IoU threshold for non-maximum suppression (0.0-1.0)

**Response Structure**:
```json
{
  "counts": [
    {
      "timestamp": 0.033,
      "frame_number": 1,
      "count": 15
    }
  ],
  "tracks_sample": [
    {
      "track_id": 1,
      "bbox": {
        "x1": 100.5,
        "y1": 200.3,
        "x2": 150.8,
        "y2": 250.6
      },
      "confidence": 0.89,
      "frame_number": 45
    }
  ],
  "weight_estimates": [
    {
      "track_id": 1,
      "weight_value": 45.3,
      "unit": "index",
      "confidence": 0.75,
      "uncertainty": 2.1,
      "method": "bbox_area_with_perspective_averaged"
    }
  ],
  "artifacts": {
    "output_video": "outputs/videos/annotated_20231218_143022.mp4",
    "output_json": "outputs/json/analysis_20231218_143022.json"
  },
  "metadata": {
    "input_video": {
      "fps": 30.0,
      "frame_count": 900,
      "width": 1920,
      "height": 1080,
      "duration_seconds": 30.0
    },
    "processing": {
      "fps_sample": 2,
      "conf_thresh": 0.5,
      "iou_thresh": 0.4
    },
    "tracking_statistics": {
      "total_unique_tracks": 25,
      "average_count": 18.5,
      "max_count": 22,
      "min_count": 14,
      "average_track_lifespan": 120.3,
      "total_frames_processed": 450
    },
    "weight_statistics": {
      "mean_weight_proxy": 42.5,
      "std_weight_proxy": 8.3,
      "min_weight_proxy": 28.1,
      "max_weight_proxy": 65.7,
      "total_samples": 2250
    }
  },
  "calibration_notes": "..."
}
```

## Implementation Details

### 1. Bird Detection

**Method**: YOLOv8 (You Only Look Once version 8)

**Approach**:
- Uses pretrained YOLOv8n (nano) model for real-time performance
- Filters for "bird" class (class 14 in COCO dataset)
- For production with poultry-specific data, train a custom model on annotated poultry images
- Configurable confidence threshold to balance precision/recall

**Handling False Positives**:
- Confidence thresholding removes low-quality detections
- Non-maximum suppression (NMS) eliminates duplicate detections
- Minimum bounding box area filter removes noise

### 2. Bird Tracking

**Method**: DeepSORT (Deep Simple Online and Realtime Tracking)

**Approach**:
- Combines motion prediction (Kalman filter) with appearance features (CNN embeddings)
- Assigns stable IDs to birds across frames
- Handles temporary occlusions by maintaining track state

**Occlusion Handling**:
- Tracks remain active for `max_age` frames without detections (default: 30 frames)
- Motion prediction estimates position during occlusion
- Re-association based on appearance similarity when bird reappears

**ID Switch Minimization**:
- Deep appearance features distinguish between similar-looking birds
- Conservative matching thresholds prevent incorrect associations
- Track confirmation requires `min_hits` detections (default: 3) before ID is assigned

**Double-Counting Prevention**:
- Each detection is assigned to at most one track
- Hungarian algorithm ensures optimal detection-to-track assignment
- Unique track IDs prevent counting the same bird multiple times

### 3. Weight Estimation

**Method**: Feature-based weight proxy with perspective correction

**Approach**:
Since ground truth weights are unavailable in typical video datasets, we calculate a **weight proxy/index** based on:

1. **Bounding Box Area**: Primary feature correlating with bird size
2. **Aspect Ratio**: Birds have characteristic width-to-height ratios
3. **Perspective Correction**: Adjusts for camera angle (birds lower in frame appear larger)
4. **Temporal Averaging**: Averages estimates across track lifetime for stability

**Formula**:
```
corrected_area = bbox_area × perspective_factor
normalized_area = corrected_area / (frame_width × frame_height)
weight_proxy = normalized_area × calibration_factor × 10000
```

**Confidence Calculation**:
- Based on aspect ratio quality (closer to 1.0 = higher confidence)
- Based on bounding box size (larger = more reliable)
- Uncertainty estimated from variance across track lifetime

**Converting to Actual Grams**:

To convert the weight proxy to actual weights in grams, you need:

1. **Reference Object Calibration**:
   - Place objects of known size in the frame
   - Calculate pixel-to-cm conversion factor
   - Use species-specific size-to-weight growth curves

2. **Ground Truth Annotation**:
   - Manually weigh sample birds
   - Record weights and video timestamps
   - Train regression model: `weight_proxy → actual_weight_grams`
   - Example: `weight_grams = a × weight_proxy + b`

3. **Camera Calibration**:
   - Perform camera calibration for intrinsic/extrinsic parameters
   - Apply proper perspective transformation
   - Account for camera height, angle, and lens distortion

4. **Species-Specific Models**:
   - Different poultry species have different size-to-weight ratios
   - Collect species-specific training data
   - Train separate models or use species as a feature

**Current Limitations**:
- Weight proxy is in arbitrary units (relative index)
- Requires calibration for absolute weight in grams
- Accuracy depends on camera setup consistency
- Assumes birds are on the same ground plane

## Output Files

### Annotated Video

- Location: `outputs/videos/annotated_YYYYMMDD_HHMMSS.mp4`
- Contains:
  - Green bounding boxes around detected birds
  - Tracking IDs and weight proxy values
  - Bird count overlay in top-left corner
- Same resolution and duration as input video

### JSON Results

- Location: `outputs/json/analysis_YYYYMMDD_HHMMSS.json`
- Contains complete analysis results (see Response Structure above)
- Can be used for further analysis, visualization, or reporting

## Configuration

Edit `src/config.py` to customize:

- Model selection (YOLOv8n/s/m/l/x)
- Detection thresholds
- Tracking parameters
- Weight estimation calibration
- Output video settings
- API settings

## Performance Optimization

### Frame Sampling

Processing every frame is computationally expensive. Use `fps_sample` parameter:
- `fps_sample=1`: Process all frames (highest accuracy, slowest)
- `fps_sample=2`: Process every other frame (recommended balance)
- `fps_sample=3`: Process every 3rd frame (faster, may miss fast-moving birds)

### Model Selection

In `src/config.py`, change `YOLO_MODEL`:
- `yolov8n.pt`: Fastest, lowest accuracy
- `yolov8s.pt`: Balanced
- `yolov8m.pt`: Better accuracy, slower
- `yolov8l.pt` / `yolov8x.pt`: Best accuracy, slowest

### GPU Acceleration

If you have a CUDA-capable GPU:
- YOLOv8 automatically uses GPU if available
- For DeepSORT, set `embedder_gpu=True` in `src/bird_tracker.py`

## Troubleshooting

### "No birds detected"

- Lower `conf_thresh` (e.g., 0.3)
- Check if video contains birds visible to camera
- Consider training custom model on poultry dataset

### "Too many false positives"

- Increase `conf_thresh` (e.g., 0.7)
- Increase `MIN_BOX_AREA` in `src/config.py`

### "Frequent ID switches"

- Increase `MIN_HITS` in `src/config.py`
- Decrease `IOU_THRESHOLD` for stricter matching
- Reduce `fps_sample` to process more frames

### "Processing too slow"

- Increase `fps_sample` (process fewer frames)
- Use smaller YOLO model (yolov8n)
- Reduce video resolution before processing

## Dataset Recommendations

For best results, use videos with:
- Fixed camera position (no panning/zooming)
- Good lighting conditions
- Clear view of birds from above
- Minimal occlusions
- Consistent background

**Suggested Kaggle Datasets**:
- Search: "poultry detection", "chicken counting", "bird tracking"
- Look for datasets with COCO-format annotations
- Prefer videos over images for tracking evaluation

## Future Improvements

1. **Custom Model Training**: Train YOLOv8 on poultry-specific dataset
2. **Weight Calibration**: Collect ground truth weights for regression model
3. **Multi-Camera Support**: Combine views from multiple cameras
4. **Real-time Streaming**: Process live CCTV feeds
5. **Behavior Analysis**: Detect feeding, drinking, abnormal behavior
6. **Database Integration**: Store results in database for trend analysis

## Author

**Krutika Borase**  
ML/AI Engineer Internship Candidate  
Kuppismart Solutions (Livestockify)

## Acknowledgments

This project was developed as part of the ML/AI Engineer Internship assessment. The implementation uses:
- **YOLOv8** by Ultralytics for object detection
- **DeepSORT** for multi-object tracking
- **FastAPI** for REST API development

---

**Note**: This system provides a weight **proxy/index**, not actual weights in grams. Calibration with ground truth data is required for accurate weight measurements.
