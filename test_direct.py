"""
Direct video processing test without API server.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.video_processor import VideoProcessor

def main():
    video_path = Path("data/sample/chickens.mp4")
    
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return
    
    print("=" * 60)
    print("Bird Counting System - Direct Test")
    print("=" * 60)
    print(f"\nProcessing: {video_path}")
    print(f"File size: {video_path.stat().st_size / (1024*1024):.2f} MB\n")
    
    # Create processor
    processor = VideoProcessor(
        conf_thresh=0.5,
        iou_thresh=0.4,
        fps_sample=2
    )
    
    # Process video
    print("Starting video processing...")
    print("This may take a few minutes...\n")
    
    try:
        results = processor.process_video(video_path)
        
        print("\n" + "=" * 60)
        print("Processing Complete!")
        print("=" * 60)
        
        # Print summary
        stats = results['metadata']['tracking_statistics']
        print(f"\n📊 Results Summary:")
        print(f"  • Total frames processed: {stats['total_frames_processed']}")
        print(f"  • Unique birds tracked: {stats['total_unique_tracks']}")
        print(f"  • Average bird count: {stats['average_count']:.1f}")
        print(f"  • Max bird count: {stats['max_count']}")
        print(f"  • Min bird count: {stats['min_count']}")
        
        print(f"\n📁 Output Files:")
        print(f"  • Video: {results['artifacts']['output_video']}")
        print(f"  • JSON: {results['artifacts']['output_json']}")
        
        print("\n✅ Success! Check the outputs folder for results.")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
