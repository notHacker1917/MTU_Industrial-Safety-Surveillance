#!/usr/bin/env python3
"""
Visual demo of the PPE tracking system.
Runs the tracker with mock pipeline and displays results with annotated frame.
"""

import cv2
import numpy as np
from tracker_ppe import TrackerPPE, Track
from oak_pipeline_mock import MockOakDPipeline
from ppe_rules import DEFAULT_ZONE_RULES

def main():
    print("=" * 60)
    print("PPE COMPLIANCE TRACKING SYSTEM - VISUAL DEMO")
    print("=" * 60)
    print()

    # Initialize pipeline and tracker
    pipeline = MockOakDPipeline(blob_path="mock")  # Mock path for mock pipeline
    tracker = TrackerPPE(
        frame_width=320,
        frame_height=320,
        ppe_classifier=None,
        zone_rules=DEFAULT_ZONE_RULES
    )
    
    frame_count = 0
    max_frames = 300
    
    print(f"Running demo for {max_frames} frames...")
    print("Press Q in window to quit early, or wait for demo to complete.")
    print()
    
    while frame_count < max_frames:
        frame_count += 1
        
        # Get frame and detections from mock pipeline
        annotated_frame, detections = pipeline.get_frame()
        
        # Update tracker
        current_zone = "A" if frame_count % 100 < 50 else "B"
        tracks, frame_meta = tracker.update(detections, current_zone, annotated_frame)
        
        # Display statistics every 30 frames
        if frame_count % 30 == 0:
            print(f"\n[FRAME {frame_count}]")
            print(f"  Active Tracks: {len(tracks)}")
            print(f"  Zone: {current_zone}")
            print(f"  People Entered: {frame_meta.get('people_entered', '?')}")
            print(f"  People Inside: {frame_meta.get('people_inside', '?')}")
            
            if tracks:
                print(f"  Sample Tracks:")
                for track in tracks[:3]:
                    compliance = track['compliance']
                    print(f"    Track #{track['track_id']}: "
                          f"[{compliance['alert_level'].upper()}] "
                          f"Depth: {track['depth_m']:.1f}m "
                          f"Compliant: {compliance['compliant']}")
        
        # Display frame
        if frame_count == 1 or frame_count % 100 == 0:
            # Add text overlay
            frame_display = annotated_frame.copy()
            cv2.putText(frame_display, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_display, f"Zone: {current_zone}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_display, f"Tracks: {len(tracks)}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("PPE Tracker Demo", frame_display)
            key = cv2.waitKey(33) & 0xFF
            if key == ord('q'):
                print("\nQuitting early...")
                break
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"Total Frames Processed: {frame_count}")
    print(f"Final Statistics:")
    print(f"  Total Tracks Created: {Track.id_counter - 1}")
    print(f"  People Entered: {frame_meta.get('people_entered', '?')}")
    print(f"  People Inside: {frame_meta.get('people_inside', '?')}")
    print(f"  Active Tracks: {len(tracks)}")
    print()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
