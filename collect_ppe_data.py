#!/usr/bin/env python3
"""
A7: Live PPE Data Collection Script
Captures frames from OAK-D RGB camera and labels for Roboflow training.
Press keys to label: S/N (suit), H/X (shield), G/B (gloves), SPACE (capture), Q (quit)
Saves to data/ppe_training/{label_combo}/ directory
"""

import cv2
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    from oak_pipeline_mock import OakDPipeline
    print("[INFO] Using mock OAK-D pipeline (hardware unavailable)")
except ImportError:
    try:
        from oak_pipeline import OakDPipeline
        print("[INFO] Using real OAK-D hardware")
    except ImportError:
        print("[ERROR] OAK-D pipeline not found")
        print("[INFO] Using OpenCV camera as fallback")
        OakDPipeline = None


class PPEDataCollector:
    """Collect PPE training data with live camera feed and keyboard labeling."""

    def __init__(self, output_dir: str = "data/ppe_training"):
        """
        Initialize collector.

        Args:
            output_dir: Root directory for training data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Label state (toggles with key presses)
        self.current_labels = {
            "suit": None,  # True/False/None
            "shield": None,
            "gloves": None,
        }

        # Counters per label combination
        self.label_counts = defaultdict(int)

        # Target: 40 images per class combination (3^3 = 27 combinations)
        self.target_per_class = 40

        # FPS tracking
        self.frame_count = 0
        self.total_captured = 0

    def get_label_combo(self) -> str:
        """Generate label directory name from current labels."""
        parts = []
        for key in ["suit", "shield", "gloves"]:
            val = self.current_labels[key]
            if val is None:
                parts.append(f"no_{key}")
            elif val:
                parts.append(key)
            else:
                parts.append(f"no_{key}")
        return "_".join(parts)

    def capture_frame(self, frame: np.ndarray) -> None:
        """Save current frame with label."""
        label_combo = self.get_label_combo()
        label_dir = self.output_dir / label_combo
        label_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"img_{timestamp}.jpg"
        filepath = label_dir / filename

        # Save frame
        cv2.imwrite(str(filepath), frame)
        self.label_counts[label_combo] += 1
        self.total_captured += 1

        print(f"[CAPTURED] {filename} → {label_combo}")

    def draw_labels_on_frame(self, frame: np.ndarray) -> None:
        """Draw current label state on frame."""
        h, w = frame.shape[:2]

        # Background panel
        cv2.rectangle(frame, (10, 10), (w - 10, 150), (50, 50, 50), -1)

        # Title
        cv2.putText(
            frame,
            "PPE Data Collection — Press Keys to Label",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Label status
        y_pos = 70
        for label_key, label_val in self.current_labels.items():
            if label_val is None:
                label_text = f"{label_key}: UNSET"
                color = (128, 128, 128)
            elif label_val:
                label_text = f"{label_key}: YES"
                color = (0, 255, 0)
            else:
                label_text = f"{label_key}: NO"
                color = (0, 0, 255)

            cv2.putText(
                frame,
                label_text,
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                1,
            )
            y_pos += 25

        # Instructions (bottom)
        instructions = [
            "S=Suit, N=No Suit | H=Shield, X=No Shield | G=Gloves, B=No Gloves",
            "SPACE=Capture | Q=Quit",
        ]
        for i, instr in enumerate(instructions):
            cv2.putText(
                frame,
                instr,
                (10, h - 50 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

        # Running counts (top-right)
        count_text = f"Captured: {self.total_captured} | Target: {self.target_per_class * 27}"
        cv2.putText(
            frame,
            count_text,
            (w - 400, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            1,
        )

    def print_summary(self) -> None:
        """Print collection summary."""
        print("\n" + "=" * 70)
        print("COLLECTION SUMMARY")
        print("=" * 70)
        print(f"\nTotal captured: {self.total_captured} images")
        print(f"Target per class: {self.target_per_class}")
        print(f"Total possible classes: 27 (3^3 label combinations)")
        print(f"Full target: {self.target_per_class * 27} images")

        print("\nBreakdown by class:")
        for label_combo in sorted(self.label_counts.keys()):
            count = self.label_counts[label_combo]
            pct = (count / self.target_per_class * 100) if self.target_per_class > 0 else 0
            status = "[OK]" if count >= self.target_per_class else "[NEED MORE]"
            print(f"  {label_combo}: {count:3d} images {pct:5.1f}% {status}")

        print("\nNext steps:")
        print("1. Upload data/ppe_training/ to Roboflow")
        print("2. Follow: roboflow_upload_guide.txt")
        print("=" * 70 + "\n")

    def run(self) -> None:
        """Main collection loop."""
        print("\n[A7] Starting PPE Data Collection")
        print("[INFO] Initializing camera...")

        # Initialize camera
        if OakDPipeline is not None:
            try:
                pipeline = OakDPipeline(blob_path="models/yolo26n.blob")
                pipeline.start()
                print("[INFO] OAK-D pipeline ready")
                use_oak = True
            except Exception as e:
                print(f"[WARNING] OAK-D failed: {e}")
                use_oak = False
        else:
            use_oak = False

        if not use_oak:
            # Fallback to OpenCV camera
            print("[INFO] Using OpenCV USB camera fallback")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("[ERROR] Could not open camera")
                return

        print("[INFO] Collection started. Press Q to quit.\n")

        try:
            while True:
                # Get frame
                if use_oak:
                    frame, _ = pipeline.get_frame()
                    if frame is None:
                        continue
                else:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    # Resize to 320x320 for consistency
                    frame = cv2.resize(frame, (320, 320))

                self.frame_count += 1

                # Draw labels on frame
                self.draw_labels_on_frame(frame)

                # Display
                cv2.imshow("A7: PPE Data Collection", frame)

                # Handle key press
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                    print("[INFO] Quit signal received")
                    break

                # Label keys (S/N for suit)
                elif key == ord('s') or key == ord('S'):
                    self.current_labels["suit"] = True
                    print(f"[LABEL] Suit = YES")
                elif key == ord('n') or key == ord('N'):
                    self.current_labels["suit"] = False
                    print(f"[LABEL] Suit = NO")

                # Label keys (H/X for shield)
                elif key == ord('h') or key == ord('H'):
                    self.current_labels["shield"] = True
                    print(f"[LABEL] Shield = YES")
                elif key == ord('x') or key == ord('X'):
                    self.current_labels["shield"] = False
                    print(f"[LABEL] Shield = NO")

                # Label keys (G/B for gloves)
                elif key == ord('g') or key == ord('G'):
                    self.current_labels["gloves"] = True
                    print(f"[LABEL] Gloves = YES")
                elif key == ord('b') or key == ord('B'):
                    self.current_labels["gloves"] = False
                    print(f"[LABEL] Gloves = NO")

                # Capture (SPACE)
                elif key == ord(' '):
                    if any(v is None for v in self.current_labels.values()):
                        print("[WARNING] All labels must be set before capturing")
                    else:
                        self.capture_frame(frame)

        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt received")

        finally:
            if use_oak:
                pipeline.stop()
            else:
                cap.release()
            cv2.destroyAllWindows()
            self.print_summary()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import numpy as np

    collector = PPEDataCollector(output_dir="data/ppe_training")
    collector.run()
