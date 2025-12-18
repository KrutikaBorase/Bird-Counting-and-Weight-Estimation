"""
Pydantic models for API request/response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")


class TrackSample(BaseModel):
    """Sample track information."""
    track_id: int = Field(..., description="Unique tracking ID")
    bbox: BoundingBox = Field(..., description="Representative bounding box")
    confidence: float = Field(..., description="Detection confidence")
    frame_number: int = Field(..., description="Frame number where this track appears")


class CountDataPoint(BaseModel):
    """Count at a specific timestamp."""
    timestamp: float = Field(..., description="Timestamp in seconds")
    frame_number: int = Field(..., description="Frame number")
    count: int = Field(..., description="Number of birds detected")


class WeightEstimate(BaseModel):
    """Weight estimation result."""
    track_id: Optional[int] = Field(None, description="Track ID if per-bird estimate")
    weight_value: float = Field(..., description="Weight in grams or proxy index")
    unit: str = Field(..., description="Unit: 'g' for grams or 'index' for proxy")
    confidence: float = Field(..., description="Confidence score (0-1)")
    uncertainty: Optional[float] = Field(None, description="Uncertainty estimate")
    method: str = Field(..., description="Estimation method used")


class AnalysisResult(BaseModel):
    """Complete analysis result from video processing."""
    counts: List[CountDataPoint] = Field(..., description="Time series of bird counts")
    tracks_sample: List[TrackSample] = Field(..., description="Sample of detected tracks")
    weight_estimates: List[WeightEstimate] = Field(..., description="Weight estimations")
    artifacts: Dict[str, str] = Field(..., description="Generated artifact paths")
    metadata: Dict[str, Any] = Field(..., description="Processing metadata")
    calibration_notes: str = Field(..., description="Notes on calibration requirements")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="OK", description="Service status")
    version: str = Field(default="1.0.0", description="API version")
