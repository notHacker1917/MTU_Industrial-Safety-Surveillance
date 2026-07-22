"""
DepthAI OAK-D Pipeline for Real-Time Person Detection with Depth
Author: Computer Vision Expert
Hardware: OAK-D camera on Raspberry Pi 5
Compute: YOLO26n inference on MyriadX VPU
"""

import cv2
import depthai as dai
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import time
import os


class OakDPipeline:
    """
    Modular OAK-D pipeline wrapper handling:
    - RGB+Depth camera initialization
    - YOLO26n person detection on VPU
    - Spatial depth extraction per detection
    - OpenCV preprocessing (CLAHE, visibility, glare detection)
    """

    def __init__(self, blob_path: str) -> None:
        """
        Initialize OAK-D pipeline configuration.

        Args:
            blob_path: Path to YOLO26n.blob compiled for MyriadX VPU
        """
        self.blob_path = blob_path
        self.device: Optional[dai.Device] = None
        self.pipeline: Optional[dai.Pipeline] = None
        
        # FPS tracking (FIX A3: improved FPS counter using frame timestamps)
        self._frame_times = deque(maxlen=30)
        self.frame_count = 0
        self.last_time = time.time()
        
        # Debug mode (FIX A3: diagnostic output for tensor inspection)
        self.debug_mode = os.getenv('DEBUG', '0') == '1'
        self._first_frame_processed = False
        
        # Detection parameters
        self.CONFIDENCE_THRESHOLD = 0.45
        self.CLASS_ID_PERSON = 0
        self.NMS_IOU_THRESHOLD = 0.5
        self.DEPTH_MAX_METERS = 6.0
        self.DEPTH_CONFIDENCE_THRESHOLD = 200
        
        # Preprocessing parameters
        self.CLAHE_CLIP_LIMIT = 2.0
        self.CLAHE_TILE_GRID = (8, 8)
        self.LOW_VISIBILITY_THRESHOLD = 35
        self.GLARE_BRIGHTNESS_THRESHOLD = 245
        self.GLARE_PIXEL_PERCENTAGE = 0.15

    def _build_pipeline(self) -> dai.Pipeline:
        """
        Build complete DepthAI pipeline with camera, depth, and inference nodes.

        Returns:
            Configured dai.Pipeline object
        """
        pipeline = dai.Pipeline()

        # ===== ColorCamera Node (RGB Input) =====
        # Initialize 320x320 RGB camera at 30 FPS
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(320, 320)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorInfo.RGB_1080P)
        cam_rgb.setInterleaved(False)
        # FIX A3: Set color order to BGR for OpenCV compatibility
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(30)

        # ===== Stereo Depth Nodes =====
        # Left mono camera
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(dai.MonoCameraProperties.SensorInfo.RGB_1080P)
        mono_left.setCamera("left")
        mono_left.setFps(30)

        # Right mono camera
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_right.setResolution(dai.MonoCameraProperties.SensorInfo.RGB_1080P)
        mono_right.setCamera("right")
        mono_right.setFps(30)

        # StereoDepth node with depth processing
        stereo_depth = pipeline.create(dai.node.StereoDepth)
        stereo_depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo_depth.setDepthAlign(dai.CameraSensorProperties.SensorInfo.RGB)
        
        # Configure depth filtering
        stereo_depth.initialProperties.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        stereo_depth.properties.setLeftRightCheck(True)  # LR_CHECK enabled
        stereo_depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        
        # Connect mono cameras to stereo depth
        mono_left.out.link(stereo_depth.left)
        mono_right.out.link(stereo_depth.right)

        # ===== Neural Network Node (YOLO26n Inference) =====
        # Load YOLO26n.blob compiled for MyriadX VPU
        # FIX A3: Fixed blob path (models/yolo26n.blob)
        nn_network = pipeline.create(dai.node.NeuralNetwork)
        nn_network.setBlobPath(self.blob_path)
        nn_network.setNumInferenceThreads(2)

        # Connect RGB camera to NN input (320x320 BGR)
        # FIX A3: Ensure camera preview is 320x320 for blob compatibility
        cam_rgb.preview.link(nn_network.input)

        # ===== Spatial Location Calculator Node =====
        # FIX A4: Calculate depth at detection centers for spatial awareness
        spatial_calc = pipeline.create(dai.node.SpatialLocationCalculator)
        spatial_calc.setBoundingBoxScaleFactor(0.5)
        spatial_calc.setDepthLowerThreshold(self.DEPTH_CONFIDENCE_THRESHOLD)

        # Connect stereo depth to spatial calculator
        stereo_depth.depth.link(spatial_calc.inputDepth)
        nn_network.spatialDetections.link(spatial_calc.inputNormalized)

        # ===== Output Queues =====
        # Queue for depth frame (aligned to RGB)
        depth_out = pipeline.create(dai.node.XLinkOut)
        depth_out.setStreamName("depth")
        stereo_depth.depth.link(depth_out.input)

        # Queue for RGB preview
        rgb_out = pipeline.create(dai.node.XLinkOut)
        rgb_out.setStreamName("rgb")
        cam_rgb.preview.link(rgb_out.input)

        # Queue for NN detections
        nn_out = pipeline.create(dai.node.XLinkOut)
        nn_out.setStreamName("nn")
        nn_network.out.link(nn_out.input)

        # Queue for spatial detections (depth per detection)
        spatial_out = pipeline.create(dai.node.XLinkOut)
        spatial_out.setStreamName("spatial_detections")
        nn_network.spatialDetections.link(spatial_out.input)

        return pipeline

    def start(self) -> None:
        """Initialize OAK-D device and start pipeline."""
        # FIX A3: Verify OAK-D device is connected
        available = dai.Device.getAllAvailableDevices()
        if not available:
            raise RuntimeError(
                "OAK-D not found. Check USB-C cable and run: "
                "sudo chmod 666 /dev/bus/usb/*/*"
            )
        
        mxid = available[0].getMxId()
        print(f"[INFO] OAK-D found: {mxid}")
        
        self.pipeline = self._build_pipeline()
        self.device = dai.Device(self.pipeline)
        print("[INFO] OAK-D device initialized and pipeline started")

    def _nms(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to filter overlapping detections.

        Args:
            detections: List of detection dicts with 'bbox' and 'confidence'
            iou_threshold: IoU threshold for suppression (default 0.5)

        Returns:
            Filtered list of detections after NMS
        """
        if len(detections) == 0:
            return []

        # Sort by confidence descending
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        keep = []

        while len(detections) > 0:
            # Keep detection with highest confidence
            current = detections.pop(0)
            keep.append(current)

            # Remove overlapping detections
            remaining = []
            for det in detections:
                iou = self._compute_iou(current["bbox"], det["bbox"])
                if iou < iou_threshold:
                    remaining.append(det)
            detections = remaining

        return keep

    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Compute Intersection over Union between two bounding boxes.

        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]

        Returns:
            IoU value between 0 and 1
        """
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        union_area = bbox1_area + bbox2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _parse_yolo_detections(self, in_det) -> List[Dict]:
        """
        FIX A3: Parse YOLO11n output tensors to extract detections.
        Handles both flat (84, 2100) and reshaped (2100, 84) formats.

        Args:
            in_det: DepthAI NN output layer containing detection tensors

        Returns:
            List of detection dicts with bbox, confidence, class_id
        """
        detections = []

        try:
            # Get raw detection tensor
            raw = in_det.getFirstLayerFp16()
            arr = np.array(raw)

            if self.debug_mode and not self._first_frame_processed:
                print(f"[DEBUG] Raw tensor shape: {arr.shape}")
                print(f"[DEBUG] Raw tensor size: {arr.size}")
                print(f"[DEBUG] First 5 values: {arr.flat[:5]}")
                self._first_frame_processed = True

            # Handle YOLO11n output format
            # Standard format: 84 values per detection, 2100 detections total
            # 84 = [x, y, w, h, conf, class_scores...]
            if arr.size == 84 * 2100:
                # Flat format: reshape to (2100, 84)
                arr = arr.reshape(2100, 84)
                if self.debug_mode:
                    print(f"[DEBUG] Detected flat format → reshaped to {arr.shape}")
            elif arr.shape[0] == 84 and arr.shape[1] == 2100:
                # Transposed format: (84, 2100) → (2100, 84)
                arr = arr.T
                if self.debug_mode:
                    print(f"[DEBUG] Detected transposed format → transposed to {arr.shape}")
            elif arr.shape[0] == 2100 and arr.shape[1] == 84:
                # Already correct format
                if self.debug_mode:
                    print(f"[DEBUG] Detected correct format: {arr.shape}")
            else:
                print(f"[WARNING] Unexpected tensor shape: {arr.shape}, skipping")
                return []

            # Extract components: cx, cy, w, h, conf, class_scores
            cx = arr[:, 0]  # Center X (normalized 0-1)
            cy = arr[:, 1]  # Center Y (normalized 0-1)
            w = arr[:, 2]   # Width (normalized 0-1)
            h = arr[:, 3]   # Height (normalized 0-1)
            conf = arr[:, 4]  # Object confidence
            class_scores = arr[:, 5:85]  # 80 class scores (COCO)

            # Get class ID and class confidence
            class_ids = np.argmax(class_scores, axis=1)
            class_confs = np.max(class_scores, axis=1)

            # Combined confidence (object * class)
            confs = conf * class_confs

            # Filter: Person class (0) + confidence > 0.40
            mask = (class_ids == 0) & (confs > self.CONFIDENCE_THRESHOLD)
            cx, cy, w, h, confs = cx[mask], cy[mask], w[mask], h[mask], confs[mask]

            # Convert xywh normalized → xyxy pixels (320x320 frame)
            W, H = 320, 320
            x1 = np.clip((cx - w / 2) * W, 0, W).astype(int)
            y1 = np.clip((cy - h / 2) * H, 0, H).astype(int)
            x2 = np.clip((cx + w / 2) * W, 0, W).astype(int)
            y2 = np.clip((cy + h / 2) * H, 0, H).astype(int)

            # Apply NMS via OpenCV
            if len(x1) > 0:
                boxes = []
                for i in range(len(x1)):
                    w_px = x2[i] - x1[i]
                    h_px = y2[i] - y1[i]
                    if w_px > 0 and h_px > 0:
                        boxes.append([x1[i], y1[i], w_px, h_px])
                    else:
                        boxes.append([x1[i], y1[i], 1, 1])  # Prevent negative dims

                conf_list = confs.tolist()
                indices = cv2.dnn.NMSBoxes(boxes, conf_list, self.CONFIDENCE_THRESHOLD, self.NMS_IOU_THRESHOLD)

                if self.debug_mode:
                    print(f"[DEBUG] Detections before NMS: {len(x1)}, after NMS: {len(indices)}")

                # Build final detection list
                keep_indices = indices.flatten() if len(indices) > 0 else []
                for idx in keep_indices:
                    detections.append({
                        "bbox": [x1[idx], y1[idx], x2[idx], y2[idx]],
                        "confidence": float(confs[idx]),
                        "class_id": 0,  # Person
                    })

        except Exception as e:
            print(f"[ERROR] Failed to parse YOLO detections: {e}")
            import traceback
            traceback.print_exc()

        return detections

    def _extract_depth_at_detection(self, depth_frame: np.ndarray, bbox: List[int]) -> Tuple[float, bool]:
        """
        FIX A4: Extract depth (Z) in meters from stereo depth map at detection center.
        Also check validity: no-return, too-far, too-close.

        Args:
            depth_frame: Depth map in millimeters (from stereo depth)
            bbox: [x1, y1, x2, y2] in pixel coordinates

        Returns:
            Tuple of (depth_m, depth_invalid)
            depth_m: Depth in meters, rounded to 2 decimals
            depth_invalid: True if depth outside valid range
        """
        x1, y1, x2, y2 = bbox

        # FIX A4: Skip if bbox is too small (unreliable depth)
        w_px = x2 - x1
        h_px = y2 - y1
        if w_px < 20 or h_px < 20:
            return 0.0, True

        # Calculate center of detection
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Clamp to frame bounds
        center_x = max(0, min(center_x, depth_frame.shape[1] - 1))
        center_y = max(0, min(center_y, depth_frame.shape[0] - 1))

        # Extract depth value in millimeters
        depth_mm = depth_frame[center_y, center_x]

        # FIX A4: Depth validity checks
        depth_invalid = (
            depth_mm <= 0 or          # No return (black pixel)
            depth_mm > 6000 or        # Beyond 6m range
            depth_mm < 200            # Closer than 20cm (stereo min)
        )

        # Convert to meters
        depth_m = round(depth_mm / 1000.0, 2) if not depth_invalid else 0.0

        return depth_m, depth_invalid

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        FIX A8: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Converts BGR → LAB, applies CLAHE to L channel, converts back to BGR.

        Args:
            frame: Input BGR frame

        Returns:
            CLAHE-enhanced BGR frame
        """
        # Convert BGR to LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.CLAHE_CLIP_LIMIT,
            tileGridSize=self.CLAHE_TILE_GRID
        )
        l_enhanced = clahe.apply(l_channel)

        # Merge back and convert to BGR
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        frame_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return frame_enhanced

    def _detect_low_visibility(self, frame: np.ndarray) -> bool:
        """
        Detect low visibility: if frame.std() < 35, return True.

        Args:
            frame: Input BGR frame

        Returns:
            True if low visibility detected, False otherwise
        """
        frame_float = frame.astype(np.float32)
        std_dev = np.std(frame_float)
        return std_dev < self.LOW_VISIBILITY_THRESHOLD

    def _detect_glare(self, frame: np.ndarray) -> bool:
        """
        Detect glare: if >15% pixels have brightness >245, return True.

        Args:
            frame: Input BGR frame

        Returns:
            True if glare detected, False otherwise
        """
        # Convert BGR to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Count pixels with brightness > 245
        glare_pixels = np.sum(gray > self.GLARE_BRIGHTNESS_THRESHOLD)
        total_pixels = gray.size

        glare_ratio = glare_pixels / total_pixels

        return glare_ratio > self.GLARE_PIXEL_PERCENTAGE

    def _preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Apply OpenCV preprocessing and compute frame flags.

        Args:
            frame: Input BGR frame

        Returns:
            Tuple of (preprocessed_frame, frame_flags_dict)
        """
        # Apply CLAHE
        frame_enhanced = self._apply_clahe(frame)

        # Detect visibility and glare flags
        low_visibility = self._detect_low_visibility(frame)
        glare = self._detect_glare(frame)

        frame_flags = {
            "low_visibility": low_visibility,
            "glare": glare,
        }

        return frame_enhanced, frame_flags

    def get_frame(self) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """
        Capture frame from OAK-D and process through full pipeline.

        Returns:
            Tuple of (annotated_bgr_frame, list_of_detection_dicts)
            Detection dict format:
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "depth_m": float,
                "depth_invalid": bool,
                "frame_flags": {"low_visibility": bool, "glare": bool}
            }
        """
        if not self.device:
            return None, []

        # Get frames from output queues
        in_rgb = self.device.getOutputQueue("rgb").get()
        in_nn = self.device.getOutputQueue("nn").get()
        in_depth = self.device.getOutputQueue("depth").get()

        # Convert depth to OpenCV format
        depth_frame = in_depth.getFrame()  # Millimeters

        # Get BGR frame (already BGR from camera)
        frame_bgr = in_rgb.getCvFrame()  # BGR format (320x320)

        # Preprocess frame (CLAHE, visibility, glare detection)
        frame_preprocessed, frame_flags = self._preprocess_frame(frame_bgr)

        # Parse YOLO detections and apply NMS
        detections = self._parse_yolo_detections(in_nn)

        # Extract depth for each detection and build output list
        detection_results = []
        for det in detections:
            bbox = det["bbox"]
            confidence = det["confidence"]

            # FIX A4: Extract depth and get validity flag
            depth_m, depth_invalid = self._extract_depth_at_detection(depth_frame, bbox)

            detection_dict = {
                "bbox": bbox,
                "confidence": confidence,
                "depth_m": depth_m,
                "depth_invalid": depth_invalid,
                "frame_flags": frame_flags,
            }

            detection_results.append(detection_dict)

        # FIX A3: Improved FPS counter using deque of frame timestamps
        self._frame_times.append(time.monotonic())
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                fps = len(self._frame_times) / elapsed
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    print(f"[FPS] {fps:.1f} FPS (samples: {len(self._frame_times)})")
        else:
            self.frame_count += 1

        return frame_preprocessed, detection_results

    def _annotate_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        FIX A9: Draw bounding boxes, labels, and depth on annotated frame.
        Green boxes for valid depth, orange for invalid.

        Args:
            frame: Input BGR frame
            detections: List of detection dicts

        Returns:
            Annotated BGR frame
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            confidence = det["confidence"]
            depth_m = det["depth_m"]
            depth_invalid = det["depth_invalid"]

            # Choose color based on depth validity
            # Green for valid depth, orange for invalid
            color = (0, 165, 255) if depth_invalid else (0, 255, 0)  # BGR format

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Create label
            if depth_invalid:
                label = f"PERSON? {confidence:.2f}"
            else:
                label = f"PERSON {depth_m}m {confidence:.2f}"

            # Draw label background and text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (x1, y1 - text_h - 6),
                (x1 + text_w + 4, y1),
                color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1
            )

        return annotated

    def stop(self) -> None:
        """Gracefully stop device and clean up resources."""
        if self.device:
            self.device.close()
            print("[INFO] OAK-D device closed")


def main() -> None:
    """
    FIX A9: Main execution block — standalone test for A3+A4 (DepthAI pipeline + depth).
    Initialize pipeline, capture frames, display output with depth annotations.
    Press Q to quit gracefully.
    """
    # FIX A3: Updated blob path to match export output
    blob_path = "models/yolo26n.blob"

    # Initialize pipeline
    pipeline = OakDPipeline(blob_path=blob_path)

    try:
        # Start device (includes device verification)
        pipeline.start()

        print("[INFO] Pipeline started. Press 'Q' to quit.")
        print("[INFO] Displaying annotated frames with person detections and depth...")
        if os.getenv('DEBUG', '0') == '1':
            print("[DEBUG] DEBUG mode enabled - tensor diagnostics will be printed")

        while True:
            # Get frame and detections
            frame_preprocessed, detections = pipeline.get_frame()

            if frame_preprocessed is None:
                print("[WARNING] Failed to get frame, retrying...")
                continue

            # Annotate frame with bboxes, confidence, depth
            frame_with_boxes = pipeline._annotate_frame(frame_preprocessed, detections)

            # Display frame
            cv2.imshow("OAK-D Pipeline — A3+A4", frame_with_boxes)

            # Print detection summary
            if detections:
                print(f"[DETECTIONS] Found {len(detections)} person(s):")
                for i, det in enumerate(detections):
                    status = "INVALID" if det["depth_invalid"] else f"{det['depth_m']}m"
                    print(f"  [{i}] Conf: {det['confidence']:.2f}, Depth: {status}")
            else:
                # Silent on no detections to avoid spam
                pass

            # Check for quit key (Q or ESC)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # 27 is ESC
                print("[INFO] Quit signal received")
                break

    except KeyboardInterrupt:
        print("[INFO] Keyboard interrupt received, shutting down...")

    except RuntimeError as e:
        print(f"[ERROR] {e}")
        print("[INFO] Make sure OAK-D is connected via USB-C")

    finally:
        # Cleanup
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete")


if __name__ == "__main__":
    main()
