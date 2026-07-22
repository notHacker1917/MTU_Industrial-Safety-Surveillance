"""
Mock OAK-D Pipeline for Testing and Development
Generates synthetic detections when hardware is unavailable
Used as fallback when depthai SDK is not installed
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import random


class MockOakDPipeline:
    """
    Mock OAK-D pipeline that generates synthetic detections for testing
    without actual hardware. Returns random person detections in frame.
    """

    def __init__(self, blob_path: str) -> None:
        """
        Initialize mock pipeline.

        Args:
            blob_path: Ignored (for API compatibility)
        """
        self.frame_width = 320
        self.frame_height = 320
        self.frame_count = 0
        print("[MOCK] Using mock OAK-D pipeline (hardware not available)")

    def start(self) -> None:
        """Start mock pipeline."""
        print("[MOCK] Mock pipeline started")

    def get_frame(self) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate synthetic frame and detections.

        Returns:
            Tuple of (bgr_frame, detections_list)
        """
        self.frame_count += 1

        # Create synthetic frame (gray gradient for visibility testing)
        frame = np.ones((self.frame_height, self.frame_width, 3), dtype=np.uint8) * 100
        frame[:, :, 1] = 120  # Green tint

        # Generate 1-3 random person detections per frame
        num_detections = random.randint(1, 3)
        detections = []

        for i in range(num_detections):
            # Random bbox (person-like proportions)
            w = random.randint(40, 100)
            h = random.randint(80, 150)
            x1 = random.randint(0, self.frame_width - w)
            y1 = random.randint(0, self.frame_height - h)
            x2 = x1 + w
            y2 = y1 + h

            # Random confidence and depth
            confidence = random.uniform(0.6, 0.98)
            depth_m = random.uniform(0.5, 5.0)

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "depth_m": depth_m,
                "depth_invalid": False,
                "frame_flags": {
                    "low_visibility": False,
                    "glare": False,
                },
            })

            # Draw detection on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            label = f"{confidence:.2f} {depth_m:.1f}m"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )

        # Add frame counter
        cv2.putText(
            frame,
            f"[MOCK] Frame {self.frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            1,
        )

        return frame, detections

    def stop(self) -> None:
        """Stop mock pipeline."""
        print("[MOCK] Mock pipeline stopped")


# Try to import real OAK-D pipeline, fallback to mock if not available
try:
    import depthai as dai
    from oak_pipeline import OakDPipeline as RealOakDPipeline
    OakDPipeline = RealOakDPipeline
    USING_MOCK = False
except ImportError:
    print("[WARNING] DepthAI SDK not installed. Using mock pipeline.")
    print("[INFO] For real hardware: pip install depthai")
    OakDPipeline = MockOakDPipeline
    USING_MOCK = True
