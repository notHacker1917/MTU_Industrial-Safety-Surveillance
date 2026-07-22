"""
Centralized configuration loader for Safety Rover.
Loads rover_params.yaml with validation and type safety.
Singleton pattern - load once, import anywhere.
"""

import os
import yaml
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG DATACLASSES (Type-safe config with validation)
# ============================================================================

@dataclass
class NetworkConfig:
    rover_pi_ip: str
    rosbridge_websocket_port: int
    cam_mjpeg_port: int
    dashboard_port: int

@dataclass
class HardwareConfig:
    lidar_port: str
    lidar_baudrate: int
    hcsr04_trigger_pin: int
    hcsr04_echo_pin: int
    hcsr04_max_distance_m: float
    ruuvi_ble_address: str
    ruuvi_read_interval_seconds: int

@dataclass
class OakDConfig:
    frame_width: int
    frame_height: int
    fps: int
    rgb_stereo_aligned: bool
    spatial_depth_median_filter: bool
    spatial_depth_left_right_check: bool
    spatial_depth_extended_mode: bool

@dataclass
class PreprocessingConfig:
    clahe_enabled: bool
    clahe_clip_limit: float
    clahe_tile_size: int
    apply_bilateral_filter: bool

@dataclass
class DetectionConfig:
    confidence_threshold: float
    nms_iou_threshold: float
    max_detections: int
    depth_max_valid_m: float
    depth_invalid_value: float
    low_visibility_std_threshold: int
    glare_detection_enabled: bool
    glare_bright_pixel_threshold: float

@dataclass
class BytetrackConfig:
    iou_match_threshold: float
    min_frames_for_confirmation: int
    max_lost_frames: int
    kalman_process_noise_position: float
    kalman_process_noise_velocity: float
    kalman_process_noise_size: float
    kalman_measurement_noise: float

@dataclass
class LineCrossingConfig:
    line_y_ratio: float
    cooldown_seconds: int

@dataclass
class PPEConfig:
    classification_interval_frames: int
    confidence_threshold: float
    suit_confidence_threshold: float
    shield_confidence_threshold: float
    gloves_confidence_threshold: float
    shield_heuristic_enabled: bool

@dataclass
class FaceBlurConfig:
    enabled: bool
    blur_kernel_size: int
    blur_sigma: float
    blur_region_ratio: float

@dataclass
class VisionConfig:
    oak_d: OakDConfig
    preprocessing: PreprocessingConfig
    detection: DetectionConfig
    bytetrack: BytetrackConfig
    line_crossing: LineCrossingConfig
    ppe: PPEConfig
    face_blur: FaceBlurConfig

@dataclass
class ZoneConfig:
    name: str
    polygon: Optional[List[List[int]]]
    ppe_requirements: List[str]
    critical_item: Optional[str]
    warning_item: Optional[str]

@dataclass
class EnvironmentThresholds:
    temp_alert_celsius: float
    temp_critical_celsius: float
    humidity_alert_pct: float
    humidity_critical_pct: float

@dataclass
class AlertsConfig:
    alert_cooldown_seconds: int
    critical_alert_cooldown_seconds: int
    environment_escalation_enabled: bool
    environment_thresholds: EnvironmentThresholds
    alert_retention_minutes: int

@dataclass
class LoggingConfig:
    level: str
    log_detections: bool
    log_tracks: bool
    log_ppe_classifications: bool
    log_frame_flags: bool
    frame_metadata_interval: int

@dataclass
class PerformanceConfig:
    max_frame_latency_ms: int
    max_memory_mb: int
    target_fps: int
    compress_frame_quality: int
    compress_frame_interval_hz: int

@dataclass
class RoverConfig:
    """Master configuration object - singleton."""
    network: NetworkConfig
    hardware: HardwareConfig
    vision: VisionConfig
    zones: Dict[str, ZoneConfig]
    alerts: AlertsConfig
    logging: LoggingConfig
    performance: PerformanceConfig

# ============================================================================
# CONFIG VALIDATION
# ============================================================================

def _validate_config(config: dict) -> None:
    """Validate configuration ranges and required fields."""
    
    # Vision thresholds
    ppe_cfg = config['vision']['ppe_classification']
    assert 0 <= ppe_cfg['confidence_threshold'] <= 1.0, \
        f"ppe confidence_threshold must be in [0, 1], got {ppe_cfg['confidence_threshold']}"
    
    assert 0 < ppe_cfg['classification_interval_frames'] < 100, \
        f"classification_interval_frames must be in (0, 100), got {ppe_cfg['classification_interval_frames']}"
    
    det_cfg = config['vision']['detection']
    assert 0 <= det_cfg['low_visibility_std_threshold'] <= 255, \
        f"low_visibility_std_threshold must be in [0, 255], got {det_cfg['low_visibility_std_threshold']}"
    
    assert 0 <= det_cfg['glare_bright_pixel_threshold'] <= 1.0, \
        f"glare_bright_pixel_threshold must be in [0, 1], got {det_cfg['glare_bright_pixel_threshold']}"
    
    # ByteTrack
    bt_cfg = config['vision']['bytetrack']
    assert 0 <= bt_cfg['iou_match_threshold'] <= 1.0, \
        f"iou_match_threshold must be in [0, 1], got {bt_cfg['iou_match_threshold']}"
    
    assert bt_cfg['max_lost_frames'] > 0, \
        f"max_lost_frames must be > 0, got {bt_cfg['max_lost_frames']}"
    
    # Zones
    for zone_name, zone_cfg in config.get('zones', {}).items():
        if zone_cfg['polygon']:
            assert len(zone_cfg['polygon']) >= 3, \
                f"Zone {zone_name} polygon must have >= 3 points, got {len(zone_cfg['polygon'])}"
    
    # Environment thresholds
    env_th = config['alerts']['environment_thresholds']
    assert env_th['temp_alert_celsius'] < env_th['temp_critical_celsius'], \
        "temp_alert_celsius must be < temp_critical_celsius"
    assert env_th['humidity_alert_pct'] < env_th['humidity_critical_pct'], \
        "humidity_alert_pct must be < humidity_critical_pct"
    
    logger.info("✓ Configuration validation passed")

# ============================================================================
# CONFIG LOADER (Singleton)
# ============================================================================

_config_instance: Optional[RoverConfig] = None

def load_config(config_path: Optional[str] = None) -> RoverConfig:
    """
    Load and parse rover_params.yaml with validation.
    
    Args:
        config_path: Path to YAML file. If None, searches relative to script.
    
    Returns:
        Singleton RoverConfig instance
    
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If configuration validation fails
    """
    global _config_instance
    
    if _config_instance is not None:
        return _config_instance
    
    # Find config file
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "rover_params.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    # Parse YAML
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        raise ValueError("Config file is empty")
    
    # Validate
    _validate_config(raw_config)
    
    # Build typed config
    try:
        net_cfg = raw_config['network']
        network = NetworkConfig(
            rover_pi_ip=net_cfg['rover_pi_ip'],
            rosbridge_websocket_port=net_cfg['rosbridge_websocket_port'],
            cam_mjpeg_port=net_cfg['cam_mjpeg_port'],
            dashboard_port=net_cfg['dashboard_port'],
        )
        
        hw_cfg = raw_config['hardware']
        hardware = HardwareConfig(
            lidar_port=hw_cfg['lidar']['port'],
            lidar_baudrate=hw_cfg['lidar']['baudrate'],
            hcsr04_trigger_pin=hw_cfg['hcsr04_ultrasonic']['trigger_pin'],
            hcsr04_echo_pin=hw_cfg['hcsr04_ultrasonic']['echo_pin'],
            hcsr04_max_distance_m=hw_cfg['hcsr04_ultrasonic']['max_distance_m'],
            ruuvi_ble_address=hw_cfg['ruuvi_tag']['ble_address'],
            ruuvi_read_interval_seconds=hw_cfg['ruuvi_tag']['read_interval_seconds'],
        )
        
        v_cfg = raw_config['vision']
        oak_cfg = v_cfg['oak_d']
        oak_d = OakDConfig(
            frame_width=oak_cfg['frame_width'],
            frame_height=oak_cfg['frame_height'],
            fps=oak_cfg['fps'],
            rgb_stereo_aligned=oak_cfg['rgb_stereo_aligned'],
            spatial_depth_median_filter=oak_cfg['spatial_depth']['median_filter'],
            spatial_depth_left_right_check=oak_cfg['spatial_depth']['left_right_check'],
            spatial_depth_extended_mode=oak_cfg['spatial_depth']['extended_mode'],
        )
        
        prep_cfg = v_cfg['preprocessing']
        preprocessing = PreprocessingConfig(
            clahe_enabled=prep_cfg['clahe_enabled'],
            clahe_clip_limit=prep_cfg['clahe_clip_limit'],
            clahe_tile_size=prep_cfg['clahe_tile_size'],
            apply_bilateral_filter=prep_cfg['apply_bilateral_filter'],
        )
        
        det_cfg = v_cfg['detection']
        det_flags_cfg = v_cfg['detection_flags']
        detection = DetectionConfig(
            confidence_threshold=det_cfg['confidence_threshold'],
            nms_iou_threshold=det_cfg['nms_iou_threshold'],
            max_detections=det_cfg['max_detections'],
            depth_max_valid_m=det_cfg['depth_max_valid_m'],
            depth_invalid_value=det_cfg['depth_invalid_value'],
            low_visibility_std_threshold=det_flags_cfg['low_visibility_std_threshold'],
            glare_detection_enabled=det_flags_cfg['glare_detection_enabled'],
            glare_bright_pixel_threshold=det_flags_cfg['glare_bright_pixel_threshold'],
        )
        
        bt_cfg = v_cfg['bytetrack']
        bytetrack = BytetrackConfig(
            iou_match_threshold=bt_cfg['iou_match_threshold'],
            min_frames_for_confirmation=bt_cfg['min_frames_for_confirmation'],
            max_lost_frames=bt_cfg['max_lost_frames'],
            kalman_process_noise_position=bt_cfg['kalman_process_noise_position'],
            kalman_process_noise_velocity=bt_cfg['kalman_process_noise_velocity'],
            kalman_process_noise_size=bt_cfg['kalman_process_noise_size'],
            kalman_measurement_noise=bt_cfg['kalman_measurement_noise'],
        )
        
        lc_cfg = v_cfg['line_crossing']
        line_crossing = LineCrossingConfig(
            line_y_ratio=lc_cfg['line_y_ratio'],
            cooldown_seconds=lc_cfg['cooldown_seconds'],
        )
        
        ppe_cfg = v_cfg['ppe_classification']
        ppe = PPEConfig(
            classification_interval_frames=ppe_cfg['classification_interval_frames'],
            confidence_threshold=ppe_cfg['confidence_threshold'],
            suit_confidence_threshold=ppe_cfg['suit_confidence_threshold'],
            shield_confidence_threshold=ppe_cfg['shield_confidence_threshold'],
            gloves_confidence_threshold=ppe_cfg['gloves_confidence_threshold'],
            shield_heuristic_enabled=ppe_cfg['shield_heuristic_enabled'],
        )
        
        fb_cfg = v_cfg['face_blurring']
        face_blur = FaceBlurConfig(
            enabled=fb_cfg['enabled'],
            blur_kernel_size=fb_cfg['blur_kernel_size'],
            blur_sigma=fb_cfg['blur_sigma'],
            blur_region_ratio=fb_cfg['blur_region_ratio'],
        )
        
        vision = VisionConfig(
            oak_d=oak_d,
            preprocessing=preprocessing,
            detection=detection,
            bytetrack=bytetrack,
            line_crossing=line_crossing,
            ppe=ppe,
            face_blur=face_blur,
        )
        
        zones = {}
        for zone_name, zone_cfg in raw_config.get('zones', {}).items():
            zones[zone_name] = ZoneConfig(
                name=zone_cfg['name'],
                polygon=zone_cfg['polygon'],
                ppe_requirements=zone_cfg['ppe_requirements'],
                critical_item=zone_cfg['critical_item'],
                warning_item=zone_cfg['warning_item'],
            )
        
        env_th_cfg = raw_config['alerts']['environment_thresholds']
        env_th = EnvironmentThresholds(
            temp_alert_celsius=env_th_cfg['temp_alert_celsius'],
            temp_critical_celsius=env_th_cfg['temp_critical_celsius'],
            humidity_alert_pct=env_th_cfg['humidity_alert_pct'],
            humidity_critical_pct=env_th_cfg['humidity_critical_pct'],
        )
        
        alerts_cfg = raw_config['alerts']
        alerts = AlertsConfig(
            alert_cooldown_seconds=alerts_cfg['alert_cooldown_seconds'],
            critical_alert_cooldown_seconds=alerts_cfg['critical_alert_cooldown_seconds'],
            environment_escalation_enabled=alerts_cfg['environment_escalation_enabled'],
            environment_thresholds=env_th,
            alert_retention_minutes=alerts_cfg['alert_retention_minutes'],
        )
        
        log_cfg = raw_config['logging']
        logging_config = LoggingConfig(
            level=log_cfg['level'],
            log_detections=log_cfg['log_detections'],
            log_tracks=log_cfg['log_tracks'],
            log_ppe_classifications=log_cfg['log_ppe_classifications'],
            log_frame_flags=log_cfg['log_frame_flags'],
            frame_metadata_interval=log_cfg['frame_metadata_interval'],
        )
        
        perf_cfg = raw_config['performance']
        performance = PerformanceConfig(
            max_frame_latency_ms=perf_cfg['max_frame_latency_ms'],
            max_memory_mb=perf_cfg['max_memory_mb'],
            target_fps=perf_cfg['target_fps'],
            compress_frame_quality=perf_cfg['compress_frame_quality'],
            compress_frame_interval_hz=perf_cfg['compress_frame_interval_hz'],
        )
        
        _config_instance = RoverConfig(
            network=network,
            hardware=hardware,
            vision=vision,
            zones=zones,
            alerts=alerts,
            logging=logging_config,
            performance=performance,
        )
        
        logger.info("✓ Configuration loaded successfully")
        return _config_instance
    
    except KeyError as e:
        raise ValueError(f"Missing required config field: {e}")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Configuration parsing error: {e}")

def get_config() -> RoverConfig:
    """Get loaded configuration (must call load_config first)."""
    if _config_instance is None:
        return load_config()
    return _config_instance

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    print(f"✓ Loaded config for rover at {cfg.network.rover_pi_ip}")
    print(f"  Vision: {cfg.vision.oak_d.frame_width}x{cfg.vision.oak_d.frame_height} @ {cfg.vision.oak_d.fps}fps")
    print(f"  Zones: {list(cfg.zones.keys())}")
    print(f"  Alerts cooldown: {cfg.alerts.alert_cooldown_seconds}s")
