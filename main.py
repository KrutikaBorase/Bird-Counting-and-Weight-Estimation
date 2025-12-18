"""
FastAPI application for bird counting and weight estimation.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import logging
from typing import Optional

from src.video_processor import VideoProcessor
from src.models import HealthResponse, AnalysisResult
from src.config import (
    API_HOST,
    API_PORT,
    INPUT_DIR,
    DEFAULT_CONF_THRESH,
    DEFAULT_IOU_THRESH,
    DEFAULT_FPS_SAMPLE,
    MAX_UPLOAD_SIZE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Bird Counting and Weight Estimation API",
    description="API for analyzing poultry farm CCTV videos to count birds and estimate weights",
    version="1.0.0"
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status response
    """
    logger.info("Health check requested")
    return HealthResponse(status="OK", version="1.0.0")


@app.post("/analyze_video")
async def analyze_video(
    video: UploadFile = File(..., description="Video file to analyze"),
    fps_sample: Optional[int] = Form(DEFAULT_FPS_SAMPLE, description="Process every Nth frame"),
    conf_thresh: Optional[float] = Form(DEFAULT_CONF_THRESH, description="Detection confidence threshold"),
    iou_thresh: Optional[float] = Form(DEFAULT_IOU_THRESH, description="IoU threshold for NMS")
):
    """
    Analyze video for bird counting and weight estimation.
    
    Args:
        video: Uploaded video file
        fps_sample: Process every Nth frame (default: 2)
        conf_thresh: Detection confidence threshold (default: 0.5)
        iou_thresh: IoU threshold for NMS (default: 0.4)
        
    Returns:
        Analysis results including counts, tracks, and weight estimates
    """
    logger.info(f"Video analysis requested: {video.filename}")
    logger.info(f"Parameters: fps_sample={fps_sample}, conf_thresh={conf_thresh}, iou_thresh={iou_thresh}")
    
    # Validate file type
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Supported formats: mp4, avi, mov, mkv"
        )
    
    # Save uploaded file
    input_path = INPUT_DIR / video.filename
    
    try:
        # Check file size
        video.file.seek(0, 2)  # Seek to end
        file_size = video.file.tell()
        video.file.seek(0)  # Reset to beginning
        
        if file_size > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE / (1024*1024):.0f} MB"
            )
        
        # Save file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        logger.info(f"Video saved to {input_path}")
        
        # Process video
        processor = VideoProcessor(
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            fps_sample=fps_sample
        )
        
        results = processor.process_video(input_path)
        
        logger.info("Video processing completed successfully")
        
        return JSONResponse(content=results)
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )
    
    finally:
        # Clean up uploaded file
        if input_path.exists():
            input_path.unlink()
            logger.info(f"Cleaned up input file: {input_path}")


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        API information
    """
    return {
        "name": "Bird Counting and Weight Estimation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze_video": "/analyze_video (POST)",
            "docs": "/docs"
        },
        "description": "Upload poultry farm CCTV videos to get bird counts and weight estimates"
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )
