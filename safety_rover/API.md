# Safety Rover API Documentation

Comprehensive interface reference for the Safety Rover vision pipeline components.

---

## OakPipeline

**File:** `oak_pipeline.py`  
**Purpose:** Core DepthAI vision pipeline with YOLO26n inference and stereo depth extraction

### Constructor
```python
class OakPipeline:
    def __init__(
        self,
        blob_path: str = "models/yolov26n_320_320.blob",
        frame_width: int = 320,
        frame_height: int = 320,
        depth_aligned: bool = True,
        spatial_detection_network_config: dict = None,
    ):
        """
        Initialize OAK-D pipeline.
        
        Args:
            blob_path: Path to YOLO .blob model (DepthAI format)
            frame_width: Input frame width (default 320)
            frame_height: Input frame height (default 320)
            depth_aligned: Align depth to RGB frame (default True)
            spatial_detection_network_config: Advanced config dict (optional)
        
        Raises:
            FileNotFoundError: If blob_path doesn't exist
            RuntimeError: If OAK-D not found or OakPipeline_mock substituted
        """
```

### Methods

#### `get_frame() -> Tuple[np.ndarray, List[Dict]]`
```python
def get_frame(self) -> Tuple[np.ndarray, List[Dict]]:
    """
    Capture and process single frame.
    
    Returns:
        Tuple of:
        - annotated_frame (np.ndarray, uint8): 320×320×3 BGR frame with bboxes
        - detections (List[Dict]): Person detections with structure:
            {
                "bbox": [x1, y1, x2, y2],          # Normalized [0-320]
                "confidence": float,                # [0, 1]
                "depth_m": float,                   # [0.5, 6.0] meters
                "class_id": int,                    # 0 = person
            }
    
    Raises:
        RuntimeError: If pipeline crashed
    """
```

### Properties
```python
@property
def frame_width(self) -> int:
    """Input frame width in pixels"""

@property
def frame_height(self) -> int:
    """Input frame height in pixels"""
```

### Preprocessing Details

**CLAHE (Contrast Limited Adaptive Histogram Equalization)**
- Clip Limit: 2.0
- Tile Grid: 8×8
- Purpose: Brighten underexposed frames without over-saturation

**Visibility Detection**
- Flag if frame std deviation < 35 (indicates low contrast/fog)
- Raises `low_visibility` flag in frame_meta

**Glare Detection**
- Flag if >15% of pixels have brightness >245
- Raises `glare` flag in frame_meta

---

## TrackerPPE

**File:** `tracker_ppe.py`  
**Purpose:** ByteTrack multi-object tracking with Kalman filter and PPE compliance evaluation

### Constructor
```python
class TrackerPPE:
    def __init__(
        self,
        frame_width: int = 320,
        frame_height: int = 320,
        ppe_classifier=None,  # Optional: PPE classifier instance
        zone_rules: dict = None,  # Optional: Zone config dict
    ):
        """
        Initialize tracker.
        
        Args:
            frame_width: Frame width for normalized coords
            frame_height: Frame height for normalized coords
            ppe_classifier: PPEClassifier instance (default: mock fallback)
            zone_rules: Dict with zone rules (default: from ppe_rules.py)
        """
```

### Methods

#### `update(detections: List[Dict], frame: np.ndarray, frame_id: int) -> Dict`
```python
def update(
    self,
    detections: List[Dict],
    frame: np.ndarray,
    frame_id: int,
) -> Dict:
    """
    Update tracker with new detections.
    
    Args:
        detections: List of detections from OakPipeline.get_frame()
        frame: Current frame (for PPE classification)
        frame_id: Current frame number (monotonically increasing)
    
    Returns:
        {
            "tracks": [
                {
                    "track_id": int,
                    "bbox": [x1, y1, x2, y2],
                    "depth_m": float,
                    "age": int,                      # Frames since creation
                    "ppe_status": {
                        "suit": bool | None,
                        "shield": bool | None,
                        "gloves": bool | None,
                    },
                    "compliance": {
                        "zone": str,                 # "A", "B", or "Transit"
                        "status": str,               # "OK", "WARNING", "CRITICAL"
                        "missing_ppe": [str],        # e.g. ["shield"]
                    },
                    "crossed_line": bool,            # True if crossed ref line this frame
                },
                ...
            ],
            "stats": {
                "people_tracked": int,
                "people_entered": int,               # Cumulative since start
                "people_exited": int,                # Cumulative since start
                "frame_id": int,
            },
        }
    """
```

#### `set_zone(zone: str) -> None`
```python
def set_zone(self, zone: str) -> None:
    """
    Set current zone (A, B, or Transit).
    Affects PPE compliance rules applied to all tracks.
    
    Args:
        zone: "A", "B", or "Transit"
    """
```

#### `reset() -> None`
```python
def reset() -> None:
    """Clear all tracks (start fresh for new video/scene)"""
```

### Internal Classes

#### `Track`
```python
class Track:
    """Represents single tracked person"""
    
    track_id: int                          # Unique ID (assigned once)
    bbox: List[float]                      # [x1, y1, x2, y2]
    depth_m: float                         # Latest depth estimate
    ppe_status: Dict[str, bool | None]    # {suit, shield, gloves}
    age: int                               # Frames since creation
    time_since_update: int                 # Frames since last detection match
    kalman_filter: KalmanFilter            # 8D constant-velocity model
    
    @property
    def is_confirmed(self) -> bool:
        """Track is confirmed if age >= 2 frames"""
    
    def update(self, detection: Dict) -> None:
        """Update with matched detection"""
    
    def predict(self) -> None:
        """Predict next bbox using Kalman filter"""
```

#### `KalmanFilter`
```python
class KalmanFilter:
    """8D constant-velocity motion model for tracking"""
    
    # State vector: [x, y, w, h, vx, vy, vw, vh]
    # x, y: bbox center
    # w, h: bbox width, height
    # v*: velocity components
    
    def __init__(self):
        """Initialize with zeros"""
    
    def predict(self) -> np.ndarray:
        """Predict next state (8D vector)"""
    
    def update(self, detection_bbox: List[float]) -> None:
        """Update with measurement"""
```

---

## PPEClassifier

**File:** `ppe_classifier.py`  
**Purpose:** TensorFlow Lite inference for PPE detection (suit/shield/gloves)

### Constructor
```python
class PPEClassifier:
    def __init__(
        self,
        model_path: str = "models/ppe_classifier_model.tflite",
        fallback_to_mock: bool = True,
    ):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to TFLite model
            fallback_to_mock: Use MockPPEClassifier if model not found
        
        Raises:
            FileNotFoundError: If model not found and fallback_to_mock=False
        """
```

### Methods

#### `classify(crop: np.ndarray) -> Dict | None`
```python
def classify(self, crop: np.ndarray) -> Dict | None:
    """
    Classify PPE items in person crop.
    
    Args:
        crop: Cropped BGR image of person (any size, min 40×40)
    
    Returns:
        {
            "suit": bool,                  # Detected
            "shield": bool,                # Detected
            "gloves": bool,                # Detected
            "suit_conf": float,            # [0, 1]
            "shield_conf": float,          # [0, 1]
            "gloves_conf": float,          # [0, 1]
        }
        OR None if crop too small
    
    Processing Pipeline:
        1. Crop must be >= 40×40 (returns None otherwise)
        2. Resize to 224×224
        3. BGR → RGB conversion
        4. Normalize to [0, 1]
        5. TFLite inference (sigmoid activation, multi-label)
        6. Shield heuristic (if conf in [0.35, 0.65])
    """
```

#### `_detect_shield_strap(crop: np.ndarray, conf: float) -> float`
```python
def _detect_shield_strap(self, crop: np.ndarray, conf: float) -> float:
    """
    Heuristic shield detection using HoughLines.
    
    Args:
        crop: Shield region crop
        conf: Current shield confidence
    
    Returns:
        Overridden confidence (0.9 if ≥2 horizontal lines detected, else conf)
    
    Heuristic Logic:
        - If 2+ horizontal lines in top 25% of crop → shield present
        - Override confidence to 0.9 (high confidence)
        - Otherwise return original confidence
    """
```

### Classes

#### `MockPPEClassifier`
```python
class MockPPEClassifier:
    """Fallback classifier when tflite_runtime unavailable"""
    
    def classify(self, crop: np.ndarray) -> Dict:
        """Return deterministic random PPE (seeded by crop hash)"""
```

---

## ROS2 Vision Node

**File:** `ros2_vision_node.py`  
**Purpose:** ROS2 node orchestrating pipeline, tracking, and alert generation

### Constructor
```python
class VisionNode(rclpy.node.Node):
    def __init__(self):
        """
        Initialize ROS2 node.
        
        Parameters (loaded from config):
            - blob_path: Path to YOLO model
            - tflite_model_path: Path to PPE classifier
            - frame_width, frame_height: Pipeline dimensions
            - detection_confidence: YOLO confidence threshold
            - nms_threshold: YOLO NMS threshold
            - bytetrack_iou_threshold: ByteTrack IoU threshold
            - ppe_confidence_threshold: PPE classifier threshold
        """
```

### Publishers
```python
# /vision/detections (15 Hz)
# sensor_msgs/PointCloud2 containing detection list
# Point fields: x, y, z (depth), intensity (confidence)

# /vision/frame_meta (15 Hz)
# Custom message with frame stats
# Fields: frame_id, low_visibility, glare, fps

# /vision/alerts (Event-driven)
# std_msgs/String with JSON alert
# Format: {"severity": "CRITICAL|WARNING|OK", "track_id": int, "reason": str}

# /vision/annotated_frame (10 Hz)
# sensor_msgs/CompressedImage with bboxes + overlays
```

### Subscriptions
```python
# /rover/zone (std_msgs/String)
# Payload: "A" | "B" | "Transit"
# Updates PPE rules for compliance

# /environment/ruuvi (std_msgs/String)
# Payload: JSON {"temp_c": float, "humidity_pct": float}
# Escalates alerts if environmental thresholds exceeded
```

### DashboardState
```python
class DashboardState:
    """Aggregates rolling 30-min stats"""
    
    def add_alert(self, track_id: int, severity: str) -> None:
        """Log alert with timestamp"""
    
    def get_stats(self) -> Dict:
        """Return rolling window stats"""
        # Returns: {alerts_last_10min, people_seen_today, etc.}
```

---

## Configuration System

**File:** `config_loader.py` + `config/rover_params.yaml`

### ConfigLoader
```python
@dataclass
class NetworkConfig:
    pi_ip: str = "192.168.1.100"
    rosbridge_port: int = 9090
    camera_stream_port: int = 5800
    dashboard_port: int = 5000

@dataclass
class VisionConfig:
    """Wraps all vision parameters"""
    oak_d: OakDConfig
    detection: DetectionConfig
    bytetrack: ByteTrackConfig
    ppe: PPEConfig
    zones: Dict[str, ZoneConfig]  # "A", "B", "Transit"

def load_config(config_path: str = "config/rover_params.yaml") -> RoverConfig:
    """Load and validate configuration from YAML"""
    # Validates ranges, required fields
    # Returns: RoverConfig singleton
    # Raises: ValueError if validation fails
```

### Usage
```python
from config.config_loader import load_config

cfg = load_config()
print(cfg.vision.oak_d.frame_width)        # 320
print(cfg.zones["A"].ppe_requirements)     # ["suit", "shield", "gloves"]
```

---

## Dashboard Backend (Flask)

**File:** `dashboard/app.py`

### RosbridgeClient
```python
class RosbridgeClient:
    """WebSocket client for ROS2 ↔ Browser communication"""
    
    def __init__(self, url: str, topics: List[str]):
        """Connect to RosBridge and subscribe to topics"""
    
    def publish(self, topic: str, message: dict) -> None:
        """Publish message to ROS2 topic"""
    
    def on_message(self, topic: str, callback: Callable) -> None:
        """Register callback for topic messages"""
```

### Flask Routes
```python
GET /                              # Serve index.html
GET /video_feed                   # MJPEG stream proxy (CORS)
POST /api/zone                    # POST {"zone": "A|B|Transit"}
SocketIO on_connect               # Browser connects
SocketIO on_message               # Browser → Server
SocketIO emit(event, data)        # Server → Browser
```

### SocketIO Events (Server → Browser)
```python
@socketio_ns.on("detection_update")
def emit_detection(data):
    """
    {
        "track_id": int,
        "bbox": [x, y, w, h] (normalized),
        "confidence": float,
        "ppe_status": {"suit": bool, "shield": bool, "gloves": bool},
        "compliance": "OK|WARNING|CRITICAL",
    }
    """

@socketio_ns.on("alert_event")
def emit_alert(data):
    """
    {
        "severity": "CRITICAL|WARNING|OK",
        "track_id": int,
        "message": str,
        "timestamp": ISO8601,
    }
    """

@socketio_ns.on("stats_update")
def emit_stats(data):
    """Every 10 seconds:
    {
        "fps": float,
        "people_tracked": int,
        "people_entered": int,
        "people_exited": int,
        "alerts_last_10min": int,
    }
    """
```

---

## Example Usage

### Example 1: Basic Vision Pipeline
```python
from oak_pipeline import OakPipeline
from tracker_ppe import TrackerPPE
from ppe_classifier import PPEClassifier

# Initialize
pipeline = OakPipeline(blob_path="models/yolov26n_320_320.blob")
classifier = PPEClassifier(model_path="models/ppe_classifier_model.tflite")
tracker = TrackerPPE(frame_width=320, frame_height=320, ppe_classifier=classifier)

# Main loop
frame_id = 0
while True:
    frame, detections = pipeline.get_frame()
    
    result = tracker.update(detections, frame, frame_id)
    
    for track in result["tracks"]:
        print(f"Person {track['track_id']}: {track['compliance']['status']}")
    
    frame_id += 1
```

### Example 2: Zone Switching
```python
tracker.set_zone("A")  # High-hazard zone
# Now requires: suit + shield + gloves

tracker.set_zone("B")  # Moderate-hazard zone
# Now requires: gloves only
```

### Example 3: ROS2 Integration
```python
# In ros2_vision_node.py
result = self.tracker.update(detections, frame, frame_id)

# Publish detections
detections_msg = self._convert_to_pointcloud(result["tracks"])
self.publisher_detections.publish(detections_msg)

# Publish alerts
for alert in result.get("alerts", []):
    alert_msg = String(data=json.dumps(alert))
    self.publisher_alerts.publish(alert_msg)
```

---

## Error Handling

All components use graceful degradation:

- **OAK-D not found** → Uses `OakPipeline_mock` (synthetic detections)
- **PPE model missing** → Uses `MockPPEClassifier` (deterministic mock)
- **RosBridge unavailable** → Dashboard shows "RosBridge offline"
- **Config invalid** → Raises `ValueError` with detailed message

---

## Performance Notes

- **Inference (YOLO):** MyriadX VPU, ~33ms @ 320×320
- **Tracking:** ByteTrack + Kalman, ~10ms per frame
- **PPE Classification:** TFLite CPU, ~40ms per person
- **Total latency:** ~83ms (2-3 frames @ 30 FPS)

---

**Last Updated:** 2026-06-16 | Version: 0.1.0
