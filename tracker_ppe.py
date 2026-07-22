"""
Multi-Object Tracker with PPE (Personal Protective Equipment) Compliance
Uses ByteTrack (IoU-based + Kalman Filter) with zone-aware PPE rules.
Privacy: Face blurring before display, no biometric storage, session-only track IDs.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import linear_sum_assignment
import time
from collections import defaultdict

from ppe_rules import (
    DEFAULT_ZONE_RULES,
    LINE_CROSSING_CONFIG,
    TRACK_CONFIG,
    DISPLAY_CONFIG,
    FACE_BLUR_CONFIG,
    KALMAN_CONFIG,
)


# ============================================================================
# KALMAN FILTER IMPLEMENTATION
# ============================================================================

class KalmanFilter:
    """
    Simple 2D Kalman Filter for tracking bounding box centroids.
    State: [cx, cy, w, h, vx, vy, vw, vh]
    """

    def __init__(self) -> None:
        """Initialize Kalman filter matrices."""
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.ndim = 8
        self.dt = KALMAN_CONFIG["dt"]

        # State transition matrix (constant velocity model)
        self.F = np.eye(self.ndim, self.ndim)
        self.F[0, 4] = self.dt  # cx += vx * dt
        self.F[1, 5] = self.dt  # cy += vy * dt
        self.F[2, 6] = self.dt  # w += vw * dt
        self.F[3, 7] = self.dt  # h += vh * dt

        # Measurement matrix (we measure cx, cy, w, h directly)
        self.H = np.eye(4, self.ndim)

        # Process noise covariance
        self.Q = np.eye(self.ndim) * KALMAN_CONFIG["process_noise_position"]
        self.Q[4:, 4:] *= KALMAN_CONFIG["process_noise_velocity"]
        self.Q[2:4, 2:4] *= KALMAN_CONFIG["process_noise_size"]

        # Measurement noise covariance
        self.R = np.eye(4) * KALMAN_CONFIG["measurement_noise"]

        # State covariance
        self.P = np.eye(self.ndim)

        # State vector
        self.x = np.zeros(self.ndim)

    def predict(self) -> np.ndarray:
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]  # Return [cx, cy, w, h]

    def update(self, measurement: np.ndarray) -> None:
        """Update state with measurement [cx, cy, w, h]."""
        z = measurement
        y = z - self.H @ self.x  # Innovation

        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(self.ndim) - K @ self.H) @ self.P
        
        # FIX A5: On second update, compute velocity from position delta
        # Check if velocity was previously zero (first update only)
        if np.allclose(self.x[4:], 0) and not np.allclose(self.x[:4], z):
            # Estimate velocity from position change
            prev_pos = (self.H @ self.x)[:4]
            vel = z - prev_pos
            # Clamp velocity: ±40 pixels/frame max to prevent runaway
            vel_clamped = np.clip(vel, -40, 40)
            self.x[4:] = vel_clamped

    def init_state(self, measurement: np.ndarray) -> None:
        """Initialize state from first measurement [cx, cy, w, h]."""
        self.x[:4] = measurement
        self.x[4:] = 0  # Zero velocity initially


# ============================================================================
# TRACK CLASS
# ============================================================================

class Track:
    """
    Single object track with Kalman filter state, history, and metadata.
    """

    # Class variable for track ID generation
    next_id = 1

    def __init__(self, detection: Dict[str, Any]) -> None:
        """
        Initialize a new track from a detection.

        Args:
            detection: Detection dict with bbox, confidence, depth_m, etc.
        """
        self.track_id = Track.next_id
        Track.next_id += 1

        self.kalman = KalmanFilter()
        self.bbox = detection["bbox"]
        self.confidence = detection["confidence"]
        self.depth_m = detection["depth_m"]

        # Extract centroid and size
        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        measurement = np.array([cx, cy, w, h], dtype=np.float32)
        self.kalman.init_state(measurement)

        # Track lifecycle
        self.age = 0  # Frames since track creation
        self.time_since_update = 0  # Frames since last matched detection
        self.is_confirmed = False  # Track confirmed after N frames

        # PPE tracking
        self.ppe_last_classification_frame = -999  # Frame when PPE was last classified
        self.ppe_cached = None  # Cached PPE classification result
        self.ppe_confidences = {}

        # Line crossing tracking (FIX A6: Add hysteresis band state)
        self.last_centroid = np.array([cx, cy])
        self.last_cross_time = 0.0  # Time of last crossing
        self.last_side = None  # "above", "below", or "band" (hysteresis zone)
        self.counted_entry = False  # Track if we've counted entry for this person
        
        # FIX A5: Position-based re-entry — track bbox size for inheritance check
        self.last_bbox_size = np.array([w, h])

    def predict(self) -> None:
        """Predict next state using Kalman filter."""
        state = self.kalman.predict()
        cx, cy, w, h = state
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        self.bbox = [int(x1), int(y1), int(x2), int(y2)]

    def update(self, detection: Dict[str, Any]) -> None:
        """
        Update track with new detection.

        Args:
            detection: Detection dict with bbox, confidence, depth_m
        """
        self.bbox = detection["bbox"]
        self.confidence = detection["confidence"]
        self.depth_m = detection["depth_m"]
        self.time_since_update = 0

        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        measurement = np.array([cx, cy, w, h], dtype=np.float32)
        self.kalman.update(measurement)

        # Update centroid for crossing detection
        self.last_centroid = np.array([cx, cy])

    def mark_missed(self) -> None:
        """Mark track as not matched in current frame."""
        self.time_since_update += 1

    def is_tracked(self) -> bool:
        """Check if track is still active (not too old)."""
        return self.time_since_update < LINE_CROSSING_CONFIG["max_track_age"]

    def get_centroid(self) -> np.ndarray:
        """Get current centroid [cx, cy]."""
        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return np.array([cx, cy])


# ============================================================================
# BYTETRACK HELPER FUNCTIONS
# ============================================================================

def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
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


def iou_distance_matrix(tracks: List[Track], detections: List[Dict]) -> np.ndarray:
    """
    Compute IoU distance matrix (1 - IoU) between tracks and detections.

    Args:
        tracks: List of active Track objects
        detections: List of detection dicts

    Returns:
        Distance matrix of shape (len(tracks), len(detections))
    """
    n_tracks = len(tracks)
    n_dets = len(detections)

    cost_matrix = np.ones((n_tracks, n_dets))

    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            iou = compute_iou(track.bbox, det["bbox"])
            cost_matrix[i, j] = 1.0 - iou  # Distance = 1 - IoU

    return cost_matrix


def hungarian_assignment(cost_matrix: np.ndarray,
                        threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    FIX A5: Hungarian algorithm for bipartite matching.
    cost_matrix must be (1.0 - IoU) format — higher cost = lower IoU.
    Cost values in [0,1]: 0=perfect match, 1=no overlap.

    Args:
        cost_matrix: Cost matrix of shape (n_tracks, n_detections)
        threshold: Cost threshold for valid assignment

    Returns:
        Tuple of (matched_track_ids, matched_det_ids, unmatched_track_ids)
    """
    if cost_matrix.size == 0:
        return np.array([]), np.array([]), np.array([])

    # Apply Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # FIX A5: Filter by threshold — reject matches with cost > threshold
    valid_matches = []
    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] < threshold:  # Lower cost = better match
            valid_matches.append((r, c))

    if len(valid_matches) == 0:
        matched_tracks = np.array([])
        matched_dets = np.array([])
        unmatched_tracks = np.arange(cost_matrix.shape[0])
    else:
        matched_pairs = np.array(valid_matches)
        matched_tracks = matched_pairs[:, 0]
        matched_dets = matched_pairs[:, 1]

        unmatched_tracks = np.setdiff1d(
            np.arange(cost_matrix.shape[0]), matched_tracks
        )

    return matched_tracks, matched_dets, unmatched_tracks


# ============================================================================
# TRACKERPPE MAIN CLASS
# ============================================================================

class TrackerPPE:
    """
    Multi-object tracker with PPE compliance monitoring.
    Combines ByteTrack (IoU + Kalman) with zone-aware PPE rules.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        ppe_classifier: Optional[Any] = None,
        zone_rules: Optional[Dict] = None,
    ) -> None:
        """
        Initialize tracker.

        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            ppe_classifier: Optional PPE classifier with classify(bgr_crop) method
            zone_rules: Optional zone rules dict (uses DEFAULT_ZONE_RULES if None)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.ppe_classifier = ppe_classifier
        self.zone_rules = zone_rules or DEFAULT_ZONE_RULES

        # Active tracks
        self.tracks: List[Track] = []
        self.frame_count = 0
        
        # FIX A5: Lost tracks for position-based re-entry (last 2 seconds)
        self.lost_tracks: List[Tuple[Track, float]] = []  # List of (track, loss_time)
        self.lost_track_retention_seconds = 2.0

        # Line crossing state (FIX A6: hysteresis band tracking)
        self.line_y = int(frame_height * LINE_CROSSING_CONFIG["line_y_normalized"])
        self.hysteresis_band = 12  # pixels either side of line
        self.people_entered = 0
        self.people_exited = 0

        # Crossing cooldown per track_id (FIX A6: 4-second cooldown)
        self.crossing_cooldown: Dict[int, float] = defaultdict(float)

    def update(
        self,
        detections: List[Dict],
        current_zone: str,
        bgr_frame: np.ndarray,
    ) -> Tuple[List[Dict], Dict]:
        """
        FIX A5: Two-stage matching with Kalman filtering.
        Stage 1: high-conf (>0.55) vs ALL tracks, IoU=0.35
        Stage 2: low-conf (0.15-0.55) vs UNMATCHED tracks only, IoU=0.50
        New tracks only from unmatched HIGH-conf detections.

        Args:
            detections: List of detection dicts from oak_pipeline
            current_zone: Current zone identifier ("A", "B", "TRANSIT", "UNKNOWN")
            bgr_frame: BGR frame for PPE classification and face blurring

        Returns:
            Tuple of (track_outputs, frame_metadata)
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Expire lost tracks (older than 2 seconds)
        self.lost_tracks = [
            (t, loss_time) for t, loss_time in self.lost_tracks
            if current_time - loss_time < self.lost_track_retention_seconds
        ]

        # Predict track positions
        for track in self.tracks:
            track.predict()

        # Separate detections by confidence (FIX A5: two-stage matching)
        high_conf_dets = [d for d in detections if d["confidence"] > 0.55]
        low_conf_dets = [d for d in detections if 0.15 <= d["confidence"] <= 0.55]

        # Stage 1: High-conf vs ALL active tracks (IoU threshold 0.35)
        high_conf_cost = iou_distance_matrix(self.tracks, high_conf_dets)
        matched_t1, matched_d1, unmatched_t1 = hungarian_assignment(
            high_conf_cost, 
            threshold=1.0 - TRACK_CONFIG["iou_match_threshold"]  # 0.65 distance = 0.35 IoU
        )

        # Update matched tracks from Stage 1
        matched_tracks_set = set()
        for track_idx, det_idx in zip(matched_t1, matched_d1):
            self.tracks[track_idx].update(high_conf_dets[int(det_idx)])
            matched_tracks_set.add(track_idx)

        # Stage 2: Low-conf vs UNMATCHED tracks only (IoU threshold 0.50)
        unmatched_tracks_list = [self.tracks[i] for i in unmatched_t1]
        if len(unmatched_tracks_list) > 0 and len(low_conf_dets) > 0:
            low_conf_cost = iou_distance_matrix(unmatched_tracks_list, low_conf_dets)
            matched_t2, matched_d2, _ = hungarian_assignment(
                low_conf_cost,
                threshold=1.0 - 0.50  # 0.50 distance = 0.50 IoU threshold
            )
            
            # Update matched tracks from Stage 2
            for local_idx, det_idx in zip(matched_t2, matched_d2):
                global_idx = unmatched_t1[local_idx]
                self.tracks[global_idx].update(low_conf_dets[int(det_idx)])
                matched_tracks_set.add(global_idx)

        # Mark unmatched tracks as missing
        for track_idx, track in enumerate(self.tracks):
            if track_idx not in matched_tracks_set:
                track.mark_missed()

        # FIX A5: Position-based re-entry — try to inherit IDs from lost tracks
        unmatched_high_conf = [high_conf_dets[i] for i in range(len(high_conf_dets))
                                if i not in matched_d1]
        
        new_tracks = []
        for det in unmatched_high_conf:
            # Check if this detection matches any lost track
            inherited_id = None
            for lost_track, loss_time in self.lost_tracks:
                lost_cx, lost_cy = lost_track.get_centroid()
                new_cx, new_cy = (det["bbox"][0] + det["bbox"][2]) / 2, \
                                (det["bbox"][1] + det["bbox"][3]) / 2
                
                # Euclidean distance
                dist = np.sqrt((new_cx - lost_cx)**2 + (new_cy - lost_cy)**2)
                
                # Size check: within ±40%
                old_w = lost_track.last_bbox_size[0]
                old_h = lost_track.last_bbox_size[1]
                new_w = det["bbox"][2] - det["bbox"][0]
                new_h = det["bbox"][3] - det["bbox"][1]
                
                size_ok = (0.6 * old_w <= new_w <= 1.4 * old_w) and \
                          (0.6 * old_h <= new_h <= 1.4 * old_h)
                
                if dist < 80 and size_ok:
                    inherited_id = lost_track.track_id
                    lost_track.bbox = det["bbox"]
                    lost_track.confidence = det["confidence"]
                    lost_track.depth_m = det["depth_m"]
                    lost_track.time_since_update = 0
                    lost_track.last_bbox_size = np.array([new_w, new_h])
                    self.tracks.append(lost_track)
                    self.lost_tracks.remove((lost_track, loss_time))
                    break
            
            if inherited_id is None:
                # Create new track
                new_track = Track(det)
                new_tracks.append(new_track)
        
        self.tracks.extend(new_tracks)

        # Remove dead tracks and move to lost_tracks (FIX A5)
        tracks_to_remove = [t for t in self.tracks if not t.is_tracked()]
        for track in tracks_to_remove:
            self.lost_tracks.append((track, current_time))
        self.tracks = [t for t in self.tracks if t.is_tracked()]

        # Confirm tracks after N frames
        for track in self.tracks:
            track.age += 1
            if track.age >= TRACK_CONFIG["min_frames_for_confirmation"] and not track.is_confirmed:
                track.is_confirmed = True

        # Process confirmed tracks
        track_outputs = []
        for track in self.tracks:
            if not track.is_confirmed:
                continue

            # Check line crossing with hysteresis (FIX A6)
            crossing_event = self._check_line_crossing(track)

            # Get PPE status
            ppe_status = self._get_ppe_status(track, bgr_frame)

            # Compute PPE compliance
            compliance = self._evaluate_compliance(ppe_status, current_zone)

            # Build track output dict
            track_dict = {
                "track_id": track.track_id,
                "bbox": track.bbox,
                "depth_m": track.depth_m,
                "centroid": track.get_centroid().tolist(),
                "ppe_status": ppe_status,
                "compliance": compliance,
                "crossing_event": crossing_event,
            }

            track_outputs.append(track_dict)

        # Compute people_inside
        people_inside = max(0, self.people_entered - self.people_exited)

        # Build frame metadata
        frame_meta = {
            "people_entered": self.people_entered,
            "people_inside": people_inside,
            "frame_flags": {},
        }

        return track_outputs, frame_meta

    def _check_line_crossing(self, track: Track) -> Optional[str]:
        """
        FIX A6: Check if track crossed the line with hysteresis band.
        Hysteresis band prevents double-counting from jitter around line.
        
        LINE_Y ± 12px = hysteresis zone. Only register crossing when:
        - Previous side != "band" AND current side != "band"
        - AND previous side != current side
        - Cooldown: 4 seconds per track

        Args:
            track: Track object

        Returns:
            "entered", "exited", or None
        """
        current_time = time.time()
        centroid = track.get_centroid()
        current_y = centroid[1]

        # Determine which side the centroid is on (FIX A6: with hysteresis)
        def get_side(y):
            if y < self.line_y - self.hysteresis_band:
                return "above"
            elif y > self.line_y + self.hysteresis_band:
                return "below"
            else:
                return "band"  # In hysteresis zone — ignore

        current_side = get_side(current_y)
        crossing_event = None

        # Check for crossing only if both sides are outside band (FIX A6)
        if (track.last_side is not None and 
            track.last_side != "band" and 
            current_side != "band" and
            track.last_side != current_side):
            
            # Check cooldown (FIX A6: 4-second cooldown)
            if current_time - self.crossing_cooldown[track.track_id] > 4.0:
                if current_side == "below" and track.last_side == "above":
                    # Upward crossing (entering from above)
                    self.people_entered += 1
                    self.crossing_cooldown[track.track_id] = current_time
                    crossing_event = "entered"
                elif current_side == "above" and track.last_side == "below":
                    # Downward crossing (exiting downward)
                    self.people_exited += 1
                    self.crossing_cooldown[track.track_id] = current_time
                    crossing_event = "exited"

        # Update last_side only if outside band (prevents band noise)
        if current_side != "band":
            track.last_side = current_side

        return crossing_event

    def _get_ppe_status(self, track: Track, bgr_frame: np.ndarray) -> Dict[str, Optional[bool]]:
        """
        Get PPE classification for track.

        Args:
            track: Track object
            bgr_frame: BGR frame for classification

        Returns:
            Dict with suit, shield, gloves (bool or None if unknown)
        """
        # Run classification only on confirmed tracks and every Nth frame
        should_classify = (
            track.age >= TRACK_CONFIG["min_frames_for_ppe_classification"]
            and (self.frame_count - track.ppe_last_classification_frame) >= TRACK_CONFIG["ppe_classification_interval"]
        )

        if should_classify and self.ppe_classifier is not None:
            # Extract person crop
            x1, y1, x2, y2 = track.bbox
            crop = bgr_frame[max(0, y1):min(bgr_frame.shape[0], y2),
                            max(0, x1):min(bgr_frame.shape[1], x2)]

            if crop.size > 0:
                try:
                    result = self.ppe_classifier.classify(crop)
                    track.ppe_cached = {
                        "suit": result.get("suit"),
                        "shield": result.get("shield"),
                        "gloves": result.get("gloves"),
                    }
                    track.ppe_confidences = result.get("confidences", {})
                    track.ppe_last_classification_frame = self.frame_count
                except Exception as e:
                    print(f"[WARNING] PPE classification failed: {e}")
                    pass

        # Return cached result or None
        if track.ppe_cached is not None:
            return track.ppe_cached
        else:
            return {"suit": None, "shield": None, "gloves": None}

    def _evaluate_compliance(self, ppe_status: Dict, current_zone: str) -> Dict:
        """
        Evaluate PPE compliance based on zone rules.

        Args:
            ppe_status: Dict with suit, shield, gloves
            current_zone: Zone identifier

        Returns:
            Compliance dict with compliant (bool), missing_ppe, alert_level, zone
        """
        zone_config = self.zone_rules.get(current_zone, self.zone_rules["UNKNOWN"])
        required_ppe = zone_config.get("required_ppe", [])
        critical_items = zone_config.get("critical_items", [])
        warning_items = zone_config.get("warning_items", [])

        # Find missing PPE
        missing_ppe = []
        for ppe_item in required_ppe:
            if ppe_status.get(ppe_item) is False:  # Explicitly False (not None)
                missing_ppe.append(ppe_item)

        # Determine alert level
        alert_level = "OK"
        if missing_ppe:
            # Check for critical items
            for item in missing_ppe:
                if item in critical_items:
                    alert_level = "CRITICAL"
                    break
            # Check for warning items (only if no critical)
            if alert_level != "CRITICAL":
                for item in missing_ppe:
                    if item in warning_items:
                        alert_level = "WARNING"
                        break

        compliant = len(missing_ppe) == 0 or len(required_ppe) == 0

        return {
            "compliant": compliant,
            "missing_ppe": missing_ppe,
            "alert_level": alert_level,
            "zone": current_zone,
        }

    def get_stats(self) -> Dict:
        """
        Get cumulative session statistics.

        Returns:
            Dict with session stats
        """
        return {
            "people_entered": self.people_entered,
            "people_exited": self.people_exited,
            "people_inside": max(0, self.people_entered - self.people_exited),
            "total_tracks_created": Track.next_id - 1,
            "active_tracks": len(self.tracks),
            "confirmed_tracks": sum(1 for t in self.tracks if t.is_confirmed),
        }

    def reset_session(self) -> None:
        """Reset session stats and track IDs for new session."""
        self.tracks.clear()
        self.lost_tracks.clear()  # FIX A5: Clear lost tracks on reset
        self.people_entered = 0
        self.people_exited = 0
        self.crossing_cooldown.clear()
        self.frame_count = 0
        Track.next_id = 1
        print("[INFO] Tracker session reset")


# ============================================================================
# ANNOTATION HELPER
# ============================================================================

def annotate_frame(
    frame: np.ndarray,
    tracks: List[Dict],
    frame_meta: Dict,
    frame_height: int,
) -> np.ndarray:
    """
    Annotate frame with tracking and PPE compliance visualization.

    Args:
        frame: BGR frame to annotate
        tracks: List of track output dicts
        frame_meta: Frame metadata dict
        frame_height: Frame height for line crossing visualization

    Returns:
        Annotated BGR frame
    """
    annotated = frame.copy()

    # Draw line-crossing line
    line_y = int(frame_height * LINE_CROSSING_CONFIG["line_y_normalized"])
    cv2.line(
        annotated,
        (0, line_y),
        (annotated.shape[1], line_y),
        DISPLAY_CONFIG["line_crossing_color"],
        2,
    )

    # Draw tracks
    for track in tracks:
        x1, y1, x2, y2 = track["bbox"]
        track_id = track["track_id"]
        depth_m = track["depth_m"]
        compliance = track["compliance"]
        ppe_status = track["ppe_status"]

        # Determine box color based on compliance
        if compliance["alert_level"] == "CRITICAL":
            color = DISPLAY_CONFIG["critical_color"]
        elif compliance["alert_level"] == "WARNING":
            color = DISPLAY_CONFIG["warning_color"]
        else:
            color = DISPLAY_CONFIG["compliant_color"]

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, DISPLAY_CONFIG["bbox_thickness"])

        # Create label
        if compliance["compliant"] or len(compliance["missing_ppe"]) == 0:
            label = f"#{track_id} OK {depth_m}m"
        else:
            missing = ", ".join(compliance["missing_ppe"])
            label = f"#{track_id} MISSING: {missing}"

        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            DISPLAY_CONFIG["font_scale"],
            DISPLAY_CONFIG["line_thickness"],
        )
        cv2.rectangle(
            annotated,
            (x1, y1 - text_h - 8),
            (x1 + text_w + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            DISPLAY_CONFIG["font_scale"],
            (0, 0, 0),
            DISPLAY_CONFIG["line_thickness"],
        )

    # Draw status overlay (bottom-left)
    people_inside = frame_meta["people_inside"]
    zone = frame_meta.get("zone", "UNKNOWN")
    overlay_text = f"IN ZONE: {zone} | COUNT: {people_inside}"
    cv2.putText(
        annotated,
        overlay_text,
        (10, annotated.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    return annotated


def blur_faces(
    frame: np.ndarray,
    tracks: List[Dict],
) -> np.ndarray:
    """
    Apply Gaussian blur to face regions for privacy.

    Args:
        frame: BGR frame to blur
        tracks: List of track dicts

    Returns:
        Frame with blurred faces
    """
    blurred = frame.copy()

    for track in tracks:
        x1, y1, x2, y2 = track["bbox"]
        height = y2 - y1

        # Face is top 30% of bbox
        face_height = int(height * FACE_BLUR_CONFIG["face_roi_height_ratio"])
        face_y2 = y1 + face_height

        # Extract face ROI
        face_roi = blurred[y1:face_y2, x1:x2]

        if face_roi.size > 0:
            # Apply Gaussian blur
            blurred_roi = cv2.GaussianBlur(
                face_roi,
                FACE_BLUR_CONFIG["blur_kernel_size"],
                FACE_BLUR_CONFIG["blur_sigma"],
            )
            # Put back
            blurred[y1:face_y2, x1:x2] = blurred_roi

    return blurred


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Try real hardware first, fallback to mock
    try:
        from oak_pipeline import OakDPipeline
    except ImportError:
        try:
            from oak_pipeline_mock import OakDPipeline
            print("[INFO] Using mock pipeline (hardware unavailable)")
        except ImportError:
            print("[ERROR] Neither oak_pipeline nor oak_pipeline_mock found")
            exit(1)

    # Configuration
    BLOB_PATH = "./yolov26n_320_320.blob"
    FRAME_WIDTH = 320
    FRAME_HEIGHT = 320
    DEMO_ZONE = "A"  # Default zone for testing

    # Initialize pipeline and tracker
    oak_pipeline = OakDPipeline(blob_path=BLOB_PATH)
    tracker = TrackerPPE(
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        ppe_classifier=None,  # Testing without PPE classifier
        zone_rules=DEFAULT_ZONE_RULES,
    )

    try:
        oak_pipeline.start()
        print("[INFO] OAK-D pipeline started")
        print("[INFO] Tracker initialized (no PPE classifier - testing mode)")
        print(f"[INFO] Demo zone set to: {DEMO_ZONE}")
        print("[INFO] Press Q to quit")

        frame_counter = 0

        while True:
            # Get frame from OAK-D
            frame_bgr, detections = oak_pipeline.get_frame()

            if frame_bgr is None:
                print("[WARNING] No frame received")
                continue

            # Update tracker
            tracks, frame_meta = tracker.update(detections, DEMO_ZONE, frame_bgr)
            frame_meta["zone"] = DEMO_ZONE

            # Blur faces for privacy
            frame_privacy = blur_faces(frame_bgr, tracks)

            # Annotate frame
            frame_annotated = annotate_frame(frame_privacy, tracks, frame_meta, FRAME_HEIGHT)

            # Display
            cv2.imshow("Tracker + PPE Compliance", frame_annotated)

            # Print stats every 30 frames
            frame_counter += 1
            if frame_counter % 30 == 0:
                print(f"\n[FRAME {frame_counter}] Tracks output:")
                for track in tracks:
                    print(f"  Track #{track['track_id']}: {track}")

                stats = tracker.get_stats()
                print(f"\n[STATS] {stats}\n")

            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                print("[INFO] Quit signal received")
                break

    except KeyboardInterrupt:
        print("[INFO] Keyboard interrupt received")

    finally:
        oak_pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete")
