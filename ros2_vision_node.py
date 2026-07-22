"""
ROS2 Vision Node for PPE Compliance Monitoring
Integrates OAK-D pipeline, multi-object tracking, and PPE classification
Publishes detection data and fuses environmental alerts
"""

import json
import time
import numpy as np
import cv2
from typing import Dict, Optional, List, Any
from collections import defaultdict

# ROS2 imports with graceful fallback
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from sensor_msgs.msg import Image, CompressedImage
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("[ERROR] ROS2 not installed. Required: ros2, geometry_msgs, sensor_msgs, cv_bridge")

# Local imports
try:
    from oak_pipeline import OakDPipeline
except ImportError:
    try:
        from oak_pipeline_mock import OakDPipeline
    except ImportError:
        OakDPipeline = None

try:
    from tracker_ppe import TrackerPPE, annotate_frame, blur_faces
    from ppe_classifier import PPEClassifier
    from ppe_rules import DEFAULT_ZONE_RULES
except ImportError as e:
    print(f"[ERROR] Failed to import local modules: {e}")
    raise


class VisionNode(Node):
    """
    ROS2 Node for vision-based PPE compliance monitoring.
    """

    def __init__(self) -> None:
        """Initialize vision node with subscriptions and publishers."""
        super().__init__("vision_node")

        # Get parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                ("blob_path", "./yolov26n_320_320.blob"),
                ("tflite_model_path", "./model.tflite"),
                ("ppe_conf_threshold", 0.65),
                ("mock_mode", False),
            ],
        )

        blob_path = self.get_parameter("blob_path").value
        tflite_model_path = self.get_parameter("tflite_model_path").value
        ppe_conf_threshold = self.get_parameter("ppe_conf_threshold").value
        mock_mode = self.get_parameter("mock_mode").value

        self.get_logger().info(f"[INIT] blob_path: {blob_path}")
        self.get_logger().info(f"[INIT] tflite_model_path: {tflite_model_path}")
        self.get_logger().info(f"[INIT] ppe_conf_threshold: {ppe_conf_threshold}")
        self.get_logger().info(f"[INIT] mock_mode: {mock_mode}")

        # Initialize pipeline and tracker
        self.mock_mode = mock_mode
        self.oak_pipeline: Optional[OakDPipeline] = None
        self.tracker: Optional[TrackerPPE] = None
        self.ppe_classifier: Optional[PPEClassifier] = None
        self.camera_offline = False
        self.camera_offline_published = False

        if OakDPipeline is None:
            self.get_logger().error(
                "[ERROR] OakDPipeline not available (depthai SDK not installed)"
            )
            self.oak_pipeline = None
            self.camera_offline = True
            return

        try:
            self.oak_pipeline = OakDPipeline(blob_path=blob_path)
            self.oak_pipeline.start()
            self.get_logger().info("[INIT] OAK-D pipeline initialized")
        except Exception as e:
            self.get_logger().error(f"[ERROR] Failed to initialize OAK-D: {e}")
            self.oak_pipeline = None
            self.camera_offline = True

        # Initialize tracker
        try:
            self.tracker = TrackerPPE(
                frame_width=320,
                frame_height=320,
                ppe_classifier=None,  # Will be set below
                zone_rules=DEFAULT_ZONE_RULES,
            )
            self.get_logger().info("[INIT] Tracker initialized")
        except Exception as e:
            self.get_logger().error(f"[ERROR] Failed to initialize tracker: {e}")
            self.tracker = None

        # Initialize PPE classifier
        try:
            self.ppe_classifier = PPEClassifier(
                model_path=tflite_model_path,
                conf_threshold=ppe_conf_threshold,
            )
            # Inject classifier into tracker
            if self.tracker:
                self.tracker.ppe_classifier = self.ppe_classifier
            self.get_logger().info("[INIT] PPE classifier initialized")
        except Exception as e:
            self.get_logger().warning(f"[WARNING] PPE classifier not loaded: {e}")
            self.ppe_classifier = None
            # Publish alert about missing model
            self._publish_alert_message(
                {
                    "status": "ppe_model_missing",
                    "error": str(e),
                }
            )

        # State
        self.current_zone: str = "UNKNOWN"
        self.env_data: Dict[str, Any] = {}
        self.cv_bridge = CvBridge()

        # Alert deduplication: {track_id: last_alert_time}
        self.alert_history: Dict[int, float] = defaultdict(float)
        self.alert_dedup_cooldown = 10.0  # seconds

        # Subscriptions
        self.zone_sub = self.create_subscription(
            String,
            "/rover/zone",
            self._zone_callback,
            10,
        )
        self.env_sub = self.create_subscription(
            String,
            "/environment/ruuvi",
            self._env_callback,
            10,
        )

        # Publishers
        self.detections_pub = self.create_publisher(String, "/vision/detections", 10)
        self.frame_meta_pub = self.create_publisher(String, "/vision/frame_meta", 10)
        self.alerts_pub = self.create_publisher(String, "/vision/alerts", 10)
        self.frame_pub = self.create_publisher(CompressedImage, "/vision/annotated_frame", 10)

        # Main pipeline timer (15 Hz)
        self.pipeline_timer = self.create_timer(1.0 / 15.0, self._pipeline_callback)

        # Frame publish timer (10 Hz compressed images)
        self.last_frame_time = time.time()
        self.frame_publish_interval = 1.0 / 10.0

        # Camera offline check timer (every 5 seconds)
        self.camera_check_timer = self.create_timer(5.0, self._check_camera_online)

        self.get_logger().info("[INIT] VisionNode initialized and running")

    def _zone_callback(self, msg: String) -> None:
        """Receive current zone from rover."""
        self.current_zone = msg.data
        self.get_logger().debug(f"[ZONE] Updated to: {self.current_zone}")

    def _env_callback(self, msg: String) -> None:
        """Receive environment data from RuuviTag."""
        try:
            self.env_data = json.loads(msg.data)
            self.get_logger().debug(f"[ENV] Updated: {self.env_data}")
        except json.JSONDecodeError:
            self.get_logger().warning(f"[WARNING] Invalid JSON in environment data: {msg.data}")

    def _pipeline_callback(self) -> None:
        """
        Main vision pipeline callback (15 Hz).
        Runs detection → tracking → PPE classification → publishing.
        """
        try:
            # Check if camera is online
            if self.oak_pipeline is None:
                return

            # Get frame from OAK-D
            try:
                frame_bgr, detections = self.oak_pipeline.get_frame()
            except Exception as e:
                self.get_logger().error(f"[ERROR] Failed to get frame: {e}")
                self.camera_offline = True
                return

            if frame_bgr is None:
                self.get_logger().warning("[WARNING] No frame received")
                self.camera_offline = True
                return

            self.camera_offline = False
            self.camera_offline_published = False

            # Update tracker
            try:
                tracks, frame_meta = self.tracker.update(
                    detections,
                    self.current_zone,
                    frame_bgr,
                )
                frame_meta["zone"] = self.current_zone
                frame_meta["frame_flags"] = {"low_visibility": False, "glare": False}

            except Exception as e:
                self.get_logger().error(f"[ERROR] Tracker update failed: {e}")
                return

            # Publish detections (every callback)
            self._publish_detections(tracks)

            # Publish frame metadata (every callback)
            self._publish_frame_meta(frame_meta)

            # Check alerts and publish if needed
            self._check_and_publish_alerts(tracks)

            # Publish annotated frame (every other callback ≈ 10 Hz)
            if time.time() - self.last_frame_time >= self.frame_publish_interval:
                self._publish_annotated_frame(frame_bgr, tracks, frame_meta)
                self.last_frame_time = time.time()

        except Exception as e:
            self.get_logger().error(f"[ERROR] Pipeline callback exception: {e}")

    def _check_camera_online(self) -> None:
        """Check camera status and publish offline alert if needed."""
        if self.camera_offline and not self.camera_offline_published:
            self._publish_alert_message({"status": "camera_offline"})
            self.camera_offline_published = True

    def _publish_detections(self, tracks: List[Dict]) -> None:
        """Publish tracked detections as JSON."""
        try:
            # Convert numpy types to native Python types
            tracks_serializable = []
            for track in tracks:
                track_copy = track.copy()
                track_copy["bbox"] = [int(x) for x in track_copy["bbox"]]
                track_copy["centroid"] = [float(x) for x in track_copy["centroid"]]
                track_copy["depth_m"] = float(track_copy["depth_m"])
                track_copy["track_id"] = int(track_copy["track_id"])

                if track_copy["ppe_status"]:
                    ppe = track_copy["ppe_status"]
                    track_copy["ppe_status"] = {
                        "suit": bool(ppe["suit"]) if ppe["suit"] is not None else None,
                        "shield": bool(ppe["shield"]) if ppe["shield"] is not None else None,
                        "gloves": bool(ppe["gloves"]) if ppe["gloves"] is not None else None,
                    }

                tracks_serializable.append(track_copy)

            msg = String()
            msg.data = json.dumps(tracks_serializable)
            self.detections_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"[ERROR] Failed to publish detections: {e}")

    def _publish_frame_meta(self, frame_meta: Dict) -> None:
        """Publish frame metadata."""
        try:
            meta_copy = frame_meta.copy()
            meta_copy["people_entered"] = int(meta_copy["people_entered"])
            meta_copy["people_inside"] = int(meta_copy["people_inside"])

            msg = String()
            msg.data = json.dumps(meta_copy)
            self.frame_meta_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"[ERROR] Failed to publish frame_meta: {e}")

    def _check_and_publish_alerts(self, tracks: List[Dict]) -> None:
        """Check PPE compliance and publish alerts."""
        current_time = time.time()

        for track in tracks:
            track_id = int(track["track_id"])
            compliance = track["compliance"]
            alert_level = compliance["alert_level"]

            # Skip OK alerts
            if alert_level == "OK":
                continue

            # Check alert deduplication
            last_alert_time = self.alert_history.get(track_id, 0)
            time_since_last = current_time - last_alert_time

            # Always send CRITICAL (no cooldown)
            # Send WARNING/OTHER only if cooldown expired
            should_send = alert_level == "CRITICAL" or time_since_last >= self.alert_dedup_cooldown

            if should_send:
                # Build alert message
                alert_msg = self._build_alert_message(track, compliance)

                # Upgrade WARNING to CRITICAL if environmental hazard
                env_hazard = self._check_env_hazard()
                if env_hazard and alert_msg["alert_level"] == "WARNING":
                    alert_msg["alert_level"] = "CRITICAL"
                    alert_msg["env_hazard"] = True

                # Publish
                self._publish_alert_message(alert_msg)

                # Update history
                self.alert_history[track_id] = current_time

    def _build_alert_message(self, track: Dict, compliance: Dict) -> Dict[str, Any]:
        """Build alert message from track and compliance data."""
        return {
            "timestamp": time.time(),
            "track_id": int(track["track_id"]),
            "alert_level": compliance["alert_level"],
            "missing_ppe": compliance["missing_ppe"],
            "zone": compliance["zone"],
            "depth_m": float(track["depth_m"]),
            "env_temp": self.env_data.get("temperature"),
            "env_humidity": self.env_data.get("humidity"),
        }

    def _check_env_hazard(self) -> bool:
        """Check if environment conditions are hazardous."""
        temp = self.env_data.get("temperature")
        humidity = self.env_data.get("humidity")

        if temp and float(temp) > 45.0:
            return True
        if humidity and float(humidity) > 85.0:
            return True

        return False

    def _publish_alert_message(self, alert_dict: Dict[str, Any]) -> None:
        """Publish alert message."""
        try:
            msg = String()
            msg.data = json.dumps(alert_dict)
            self.alerts_pub.publish(msg)

            self.get_logger().warn(f"[ALERT] Published: {alert_dict}")

        except Exception as e:
            self.get_logger().error(f"[ERROR] Failed to publish alert: {e}")

    def _publish_annotated_frame(
        self,
        frame_bgr: np.ndarray,
        tracks: List[Dict],
        frame_meta: Dict,
    ) -> None:
        """Publish annotated frame as CompressedImage."""
        try:
            # Blur faces for privacy
            frame_blurred = blur_faces(frame_bgr, tracks)

            # Annotate
            frame_annotated = annotate_frame(
                frame_blurred,
                tracks,
                frame_meta,
                frame_bgr.shape[0],
            )

            # Compress
            _, buffer = cv2.imencode(".jpg", frame_annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])

            # Create ROS2 message
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = buffer.tobytes()

            self.frame_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"[ERROR] Failed to publish frame: {e}")

    def destroy_node(self) -> None:
        """Cleanup on node shutdown."""
        self.get_logger().info("[SHUTDOWN] Cleaning up...")
        if self.oak_pipeline:
            self.oak_pipeline.stop()
        super().destroy_node()


def main(args=None) -> None:
    """Entry point for ROS2 node."""
    if not ROS2_AVAILABLE:
        print("[ERROR] ROS2 not available")
        return

    rclpy.init(args=args)
    node = VisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
