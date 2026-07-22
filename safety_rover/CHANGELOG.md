# Changelog

All notable changes to Safety Rover are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Shield heuristic with HoughLines detection for uncertain confidence ranges

### Changed
- Improved CLAHE preprocessing parameters (clipLimit 2.0 → adaptive)

### Fixed
- False alerts in low-light conditions (improved visibility detection)

---

## [0.1.0] - 2026-06-16 (Pre-Demo)

### Added
- Core DepthAI OAK-D vision pipeline (oak_pipeline.py)
- ByteTrack multi-object tracking with Kalman filter (tracker_ppe.py)
- TensorFlow Lite PPE classification with mobile model (ppe_classifier.py)
- ROS2 Jazzy integration node (ros2_vision_node.py)
- Flask dashboard with real-time SocketIO updates (dashboard/)
- Type-safe configuration system with YAML (config_loader.py, rover_params.yaml)
- Idempotent Pi setup script (setup.sh)
- Master launch orchestration (launch_all.sh)
- Comprehensive unit test suite (tests/)
- Team collaboration guide (CONTRIBUTING.md)

### Infrastructure
- ROS2 packages for vision and navigation
- Model management system (download_models.sh)
- Docker support (TODO)
- CI/CD pipeline (TODO)

### Known Limitations
- Navigation stack placeholder (Person B to implement)
- Dashboard CORS testing needed for cross-origin requests
- Shield heuristic threshold tuning in progress
- Line-crossing counter may double-count at zone boundaries

---

## Release Notes

### v0.1.0 Pre-Demo (2026-06-16)
**Status:** Ready for hardware testing and live demo
- All vision components working in simulation
- Dashboard UI complete and responsive
- Team infrastructure in place (branching, review process)
- 60+ unit tests passing
- Deployment automation ready for Raspberry Pi

**Next Steps:**
- Integrate Person B's navigation stack
- Fine-tune PPE classifier thresholds on Pi
- Add telemetry logging for performance metrics
- Prepare demo scenario walk-through

---

## Versioning Notes

- **Major (X.0.0):** Major architectural changes or backward-incompatible config changes
- **Minor (0.X.0):** New features or new PPE rules/zones (backward compatible)
- **Patch (0.0.X):** Bug fixes, performance improvements, documentation

---

## How to Update This File

When merging a feature branch, add your changes under `[Unreleased]` section:

```markdown
### Added (new features for end users)
### Changed (changes in existing functionality)
### Deprecated (soon-to-be removed features)
### Removed (now removed features)
### Fixed (bug fixes)
### Security (security fixes or improvements)
```

Keep entries concise and user-focused. Link to related issues/PRs when possible.
