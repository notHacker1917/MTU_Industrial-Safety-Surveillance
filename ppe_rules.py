"""
PPE (Personal Protective Equipment) Compliance Rules Configuration
Defines zone-specific safety requirements and alert thresholds.
"""

# Default zone rules: defines required PPE for each zone
DEFAULT_ZONE_RULES = {
    "A": {
        "name": "Tank Zone (High Risk)",
        "required_ppe": ["suit", "shield", "gloves"],
        "critical_items": ["shield"],  # Missing these → CRITICAL alert
        "warning_items": ["gloves"],   # Missing these → WARNING alert
    },
    "B": {
        "name": "Walkway (Medium Risk)",
        "required_ppe": ["gloves"],
        "critical_items": [],
        "warning_items": ["gloves"],
    },
    "TRANSIT": {
        "name": "Transit Area (Low Risk)",
        "required_ppe": [],
        "critical_items": [],
        "warning_items": [],
    },
    "UNKNOWN": {
        "name": "Unknown Zone",
        "required_ppe": [],
        "critical_items": [],
        "warning_items": [],
    },
}

# Line crossing configuration
LINE_CROSSING_CONFIG = {
    "line_y_normalized": 0.55,  # 55% down from top
    "cooldown_seconds": 5.0,    # Ignore re-crossings within this time
    "max_track_age": 15,        # Max frames to keep lost track alive
}

# Track management configuration
TRACK_CONFIG = {
    "iou_match_threshold": 0.3,          # IoU threshold for association
    "min_frames_for_confirmation": 2,   # Frames needed to confirm track
    "min_frames_for_ppe_classification": 3,  # Frames before running PPE check
    "ppe_classification_interval": 5,   # Run PPE classification every N frames
}

# Display configuration
DISPLAY_CONFIG = {
    "compliant_color": (0, 255, 0),      # GREEN (BGR)
    "warning_color": (0, 255, 255),      # YELLOW (BGR)
    "critical_color": (0, 0, 255),       # RED (BGR)
    "line_crossing_color": (0, 255, 255),  # YELLOW (BGR)
    "font": "HERSHEY_SIMPLEX",
    "font_scale": 0.6,
    "line_thickness": 2,
    "bbox_thickness": 2,
}

# Face blur configuration
FACE_BLUR_CONFIG = {
    "face_roi_height_ratio": 0.30,      # Top 30% of bbox is face
    "blur_kernel_size": (99, 99),       # Gaussian blur kernel
    "blur_sigma": 30,                   # Sigma for Gaussian blur
}

# Kalman filter configuration
KALMAN_CONFIG = {
    "dt": 1.0,  # Time step (assuming 30 FPS ≈ 0.033s, but we use normalized dt=1)
    "process_noise_position": 0.1,      # Process noise for position
    "process_noise_velocity": 0.01,     # Process noise for velocity
    "process_noise_size": 0.1,          # Process noise for size
    "measurement_noise": 1.0,           # Measurement noise (detection uncertainty)
}
