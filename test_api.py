#!/usr/bin/env python3
"""
Test script for the bird counting API.
"""

import requests
import json
import sys
from pathlib import Path


def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get("http://localhost:8000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.status_code == 200


def test_analyze_video(video_path: str):
    """Test video analysis endpoint."""
    print(f"Testing /analyze_video endpoint with {video_path}...")
    
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return False
    
    with open(video_path, 'rb') as f:
        files = {'video': f}
        data = {
            'fps_sample': 2,
            'conf_thresh': 0.5,
            'iou_thresh': 0.4
        }
        
        response = requests.post(
            "http://localhost:8000/analyze_video",
            files=files,
            data=data
        )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\nAnalysis Summary:")
        print(f"- Total frames processed: {result['metadata']['tracking_statistics']['total_frames_processed']}")
        print(f"- Unique tracks: {result['metadata']['tracking_statistics']['total_unique_tracks']}")
        print(f"- Average count: {result['metadata']['tracking_statistics']['average_count']:.1f}")
        print(f"- Max count: {result['metadata']['tracking_statistics']['max_count']}")
        print(f"- Output video: {result['artifacts']['output_video']}")
        print(f"- Output JSON: {result['artifacts']['output_json']}")
        
        # Save full response
        output_file = "test_response.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nFull response saved to: {output_file}")
        
        return True
    else:
        print(f"Error: {response.text}")
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("Bird Counting API Test Script")
    print("=" * 60 + "\n")
    
    # Test health endpoint
    if not test_health():
        print("Health check failed! Is the server running?")
        print("Start server with: python main.py")
        sys.exit(1)
    
    # Test video analysis
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("Usage: python test_api.py <path_to_video>")
        print("\nExample:")
        print("  python test_api.py data/sample/poultry_video.mp4")
        sys.exit(0)
    
    if test_analyze_video(video_path):
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Video analysis failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
