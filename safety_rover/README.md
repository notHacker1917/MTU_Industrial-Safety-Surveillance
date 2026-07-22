# Safety Rover: PPE Compliance Monitoring Robot

**Real-time vision-based PPE detection and zone compliance monitoring for industrial safety.**

A ROS2-based autonomous robot system combining DepthAI OAK-D vision, ByteTrack multi-object tracking, and TensorFlow Lite PPE classification to monitor worker compliance with safety equipment requirements across designated work zones.

---

## Hardware Architecture

| Component | Model | Purpose | Notes |
|-----------|-------|---------|-------|
| **SBC** | Raspberry Pi 5 (8GB RAM) | Main compute + ROS2 orchestration | Ubuntu 24.04 LTS |
| **Camera** | OAK-D Pro | RGB-D vision pipeline (stereo depth) | DepthAI SDK, 30 FPS |
| **VPU** | Myriad X (on-camera) | YOLO inference acceleration | 320×320 @ 30 Hz |
| **LIDAR** | RPLidar A1M8 | 2D SLAM and nav | 12m range, 360° scan |
| **Ultrasonic** | HCSR04 (4×) | Obstacle detection | ~2m range |
| **Environmental** | RuuviTag Pro | Temp/humidity via BLE | Escalates PPE alerts |
| **Motor** | 2× geared DC + servo | Differential drive + turret | ROS2 Twist commands |

---

## Quick Start

### Prerequisites
- Ubuntu 24.04 on Raspberry Pi 5 (or WSL2 for dev/testing)
- 40GB free disk space
- Internet connection (for initial setup)

### Installation & Launch (4 commands)
```bash
# 1. Clone and setup environment
git clone <repo-url> safety_rover && cd safety_rover
bash setup.sh

# 2. Build ROS2 packages
cd ros2_ws && colcon build && cd ..

# 3. Download models (requires GDRIVE_ID env var)
export GDRIVE_ID="your-google-drive-folder-id"
bash models/download_models.sh

# 4. Launch everything (nodes + dashboard)
bash launch_all.sh &              # In one terminal
cd dashboard && bash start_dashboard.sh  # In another terminal
# Dashboard: http://localhost:5000
```

### Immediate Testing (No Hardware)
```bash
# Run visual demo with synthetic detections
python demo_visual.py

# Run unit tests (no OAK-D required)
cd tests && pytest -v
```

---

## Team Ownership & Responsibilities

| Person | Role | Owns | Interface |
|--------|------|------|-----------|
| **A** | Vision Lead | `/vision/*` topics, PPE classification, dashboard UI | Publishes `/vision/detections`, `/vision/alerts` |
| **B** | Navigation Lead | SLAM, Nav2, autonomous routing | Subscribes `/rover/zone`, publishes `/odom` |
| **C** | Presentation/Integration | Launch orchestration, team coordination, demos | Maintains `launch_all.sh`, docs |

---

## ROS2 Topic Map

### Published Topics (Vision Node)
| Topic | Type | Rate | Purpose |
|-------|------|------|---------|
| `/vision/detections` | `sensor_msgs/PointCloud2` | 15 Hz | Person bboxes + depth (for tracking) |
| `/vision/frame_meta` | Custom msg | 15 Hz | Frame stats (brightness, glare flags) |
| `/vision/alerts` | `std_msgs/String` | Event | PPE violations + severity (CRITICAL/WARNING/OK) |
| `/vision/annotated_frame` | `sensor_msgs/CompressedImage` | 10 Hz | Annotated frame w/ bboxes + face blur (optional) |

### Subscribed Topics (Vision Node)
| Topic | Type | Purpose |
|-------|------|---------|
| `/rover/zone` | `std_msgs/String` | Current zone (A/B/Transit) → switches PPE rules |
| `/environment/ruuvi` | `std_msgs/String` | JSON: `{temp_c, humidity_pct}` → escalates alerts |

### Navigation Topics (Person B)
| Topic | Type | Purpose |
|-------|------|---------|
| `/odom` | `nav_msgs/Odometry` | Odometry from wheel encoders |
| `/map` | `nav_msgs/OccupancyGrid` | SLAM-generated map |

---

## Dashboard Access

**URL:** `http://<pi-ip>:5000` (default: `http://localhost:5000`)

### Features
- **Live Camera Feed** with real-time detection overlays
- **Alert Log** (timestamped, scrolling)
- **Compliance Status** (person-by-person: ✓ OK / ⚠ WARNING / ✗ CRITICAL)
- **Zone Heatmap** and people counter
- **Stats Panel** (frame rate, tracked people, recent alerts)
- **Connection Indicator** (RosBridge status)

### Backend
- Flask + Flask-SocketIO (eventlet async)
- WebSocket bridge to RosBridge (ROS2 ↔ Browser)
- MJPEG proxy for camera stream (CORS-enabled)

---

## Vision Pipeline Architecture

```
OAK-D Camera (RGB-D @ 30 Hz)
    ↓
[Preprocessing: CLAHE, bilateral filter, visibility/glare detection]
    ↓
[YOLO26n @ 320×320 (MyriadX VPU inference)]
    ↓
[Stereo Depth Extraction (0.5–6.0m valid range)]
    ↓
[ByteTrack: IoU + Kalman filter (8D constant-velocity model)]
    ↓
[PPE Classification per track (TFLite MobileNetV3-Small)]
    ↓
[Zone Rules Evaluation (Zone A/B/Transit → compliance status)]
    ↓
[Line-Crossing Counter + Alert Deduplication]
    ↓
ROS2 Topics: /vision/detections, /vision/alerts, /vision/annotated_frame
```

### Key Parameters (in `config/rover_params.yaml`)
- **Detection Confidence:** 0.45 (NMS: 0.5)
- **ByteTrack IoU Threshold:** 0.3
- **PPE Confidence Threshold:** 0.65
- **Alert Cooldown:** 10s per person (3s for CRITICAL)
- **Line Crossing Cooldown:** 5s (prevents double-counting)
- **Stereo Depth Confidence:** HIGH (MyriadX)

---

## Zone Compliance Rules

### Zone A (High-Hazard Operations)
- **Required:** Suit + Shield + Gloves
- **Critical Item:** Shield (CRITICAL if missing)
- **Warning Item:** Gloves (WARNING if only missing)
- **Logic:** Suit + Gloves present but no shield → **CRITICAL** alert

### Zone B (Moderate-Hazard)
- **Required:** Gloves (minimum)
- **Warning Item:** Gloves
- **Logic:** No gloves → **WARNING**

### Transit Zone
- **Required:** None
- **Status:** Always **OK** (compliance gates open)

### Environment Escalation
- **Temp > 45°C** → Escalate WARNING items to CRITICAL
- **Humidity > 85%** → Same escalation
- **Temp > 50°C OR Humidity > 95%** → Force ALL alerts to CRITICAL

---

## Testing

### Unit Tests (No Hardware Required)
```bash
cd tests
pytest -v test_oak_pipeline.py     # Preprocessing, CLAHE, visibility/glare
pytest -v test_tracker_ppe.py      # ByteTrack, PPE rules, line-crossing
pytest -v test_ppe_classifier.py   # TFLite preprocessing, mock fallback
pytest -v                          # Run all
```

**Coverage:** ~60 test cases across preprocessing, tracking, compliance, alert dedup, environment escalation.

### Demo Playback
```bash
python demo_visual.py
# Outputs: 300 synthetic frames with zone switching, people enter/exit counts, compliance status
```

### Live Testing (Requires OAK-D + Pi)
```bash
bash launch_all.sh
# Monitor: tail -f logs/vision.log
# Test zone switching: ros2 topic pub /rover/zone std_msgs/String "data: A"
```

---

## Known Issues & Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| OAK-D not found | USB not connected / udev rules missing | Check `lsusb \| grep 03e7`, run `setup.sh` |
| tflite_runtime not available | Package not installed | Falls back to mock classifier (dev/testing OK) |
| Dashboard blank (RosBridge not responding) | RosBridge not started or network issue | Check `ros2 node list`, verify Pi network |
| False PPE alerts (shield heuristic) | Poor lighting or reflection | Retrain or adjust shield confidence threshold (config) |
| Line-crossing counter stuck | Track ID not incrementing properly | Check ByteTrack IoU threshold (should be 0.3) |
| High CPU usage on Pi | Too many tracks or YOLO resolution too high | Reduce frame resolution (config: 320×240 instead of 320×320) |

---

## Development Workflow

### Git Branching Strategy
```
main (stable, demo-ready)
├── dev-vision (Person A: pipeline, PPE, dashboard)
├── dev-nav (Person B: SLAM, Nav2, autonomous routing)
└── dev-presentation (Person C: launch, integration, docs)
```

### Git Aliases (Add to `.gitconfig`)
```bash
# Team members run these for convenient syncing
git config --global alias.sync-vision 'fetch origin && merge origin/dev-vision'
git config --global alias.sync-nav 'fetch origin && merge origin/dev-nav'
git config --global alias.sync-all 'fetch origin && merge origin/dev-vision && merge origin/dev-nav'
```

### Merge Protocol
- **Vision changes:** Person A reviews + approves (can self-merge)
- **Navigation changes:** Person B reviews (can self-merge)
- **Config changes (`config/`, `launch_all.sh`):** Require **dual review** (A + B)
- **Conflicts:** Assign to config owner, resolve via async comments

### Conflict Resolution
1. Config conflicts → Person C arbitrates with A + B input
2. Keep `rover_params.yaml` versioned; never force-push
3. Tag releases: `v1.0-demo`, `v1.1-improved-ppu`, etc.

---

## File Structure

```
safety_rover/
├── oak_pipeline.py                  # DepthAI + YOLO pipeline
├── tracker_ppe.py                   # ByteTrack + PPE compliance
├── ppe_classifier.py                # TFLite PPE classifier
├── ppe_rules.py                     # Zone rules config
├── demo_visual.py                   # 300-frame demo
├── ros2_vision_node.py              # ROS2 node (main entry)
├── config/
│   ├── rover_params.yaml            # Master configuration (250+ lines)
│   └── config_loader.py             # Type-safe YAML loader
├── models/
│   ├── download_models.sh           # Google Drive downloader (gdown)
│   └── README.md                    # Model specs & training guide
├── dashboard/
│   ├── app.py                       # Flask + RosBridge + SocketIO
│   ├── index.html                   # Single-page HTML5/CSS/JS UI
│   ├── requirements.txt
│   └── start_dashboard.sh
├── ros2_ws/
│   └── src/
│       ├── vision_pkg/              # ROS2 package (vision_node entry point)
│       └── navigation_pkg/          # ROS2 package placeholder (Person B)
├── tests/
│   ├── test_oak_pipeline.py         # Preprocessing, CLAHE, visibility
│   ├── test_tracker_ppe.py          # ByteTrack, PPE rules, line-crossing
│   ├── test_ppe_classifier.py       # TFLite preprocessing, heuristics
│   └── conftest.py                  # Pytest configuration
├── setup.sh                         # Idempotent Pi setup (Ubuntu 24.04)
├── launch_all.sh                    # Master orchestration (all nodes)
├── .gitignore
└── README.md (this file)
```

---

## Performance Characteristics

- **Frame Processing:** 30 FPS (camera) → 15 Hz detections (on-device YOLO + depth)
- **Tracking Latency:** ~33ms (ByteTrack + Kalman @ 30 FPS)
- **PPE Classification:** ~40ms per person (batched TFLite on CPU)
- **Dashboard Update:** 10 Hz (SocketIO, RosBridge)
- **Pi CPU Usage:** ~45–55% (YOLO on VPU, tracking on CPU)
- **Memory Footprint:** ~280MB (ROS2 + models + pipeline)

---

## References & Resources

- **DepthAI SDK:** https://docs.luxonis.com/en/latest/
- **ByteTrack Paper:** https://arxiv.org/abs/2110.06864
- **ROS2 Jazzy:** https://docs.ros.org/en/jazzy/
- **TensorFlow Lite:** https://www.tensorflow.org/lite
- **MobileNetV3:** https://arxiv.org/abs/1905.02175

---

## Contact & Support

- **Team Lead:** Person C
- **Vision Queries:** Person A (PPE, dashboard, detection issues)
- **Navigation Queries:** Person B (SLAM, autonomous routing)
- **Repo Issues:** Create issue with `[vision]`, `[nav]`, or `[integration]` tag

---

**Last Updated:** 2026-06-16 | **Version:** 0.1.0 (Pre-Demo)
