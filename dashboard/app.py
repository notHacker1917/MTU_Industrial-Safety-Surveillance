"""
Real-Time PPE Compliance Dashboard Backend
Flask + SocketIO server that bridges rosbridge to browser clients
"""

import os
import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import requests
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import websocket

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

ROVER_PI_IP = os.environ.get("ROVER_PI_IP", "192.168.1.100")
ROVER_CAM_PORT = os.environ.get("ROVER_CAM_PORT", "5800")
ROVER_ROSBRIDGE_PORT = os.environ.get("ROVER_ROSBRIDGE_PORT", "9090")

ROSBRIDGE_URL = f"ws://{ROVER_PI_IP}:{ROVER_ROSBRIDGE_PORT}"
MJPEG_STREAM_URL = f"http://{ROVER_PI_IP}:{ROVER_CAM_PORT}/stream"

logger.info(f"[CONFIG] Pi IP: {ROVER_PI_IP}")
logger.info(f"[CONFIG] Rosbridge: {ROSBRIDGE_URL}")
logger.info(f"[CONFIG] MJPEG Stream: {MJPEG_STREAM_URL}")

# ============================================================================
# FLASK + SOCKETIO SETUP
# ============================================================================

app = Flask(__name__, template_folder=".", static_folder="static")
app.config["SECRET_KEY"] = "dashboard-secret-key-" + ROVER_PI_IP
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ============================================================================
# GLOBAL STATE
# ============================================================================

class DashboardState:
    """Manages dashboard state and statistics."""

    def __init__(self):
        self.pi_connected = False
        self.camera_ok = False
        self.current_zone = "UNKNOWN"
        self.people_inside = 0
        self.people_entered = 0
        self.low_visibility = False
        self.glare = False

        # Event history (30-minute rolling window)
        self.events_deque = deque(maxlen=1800)  # 30 min @ 1 Hz aggregation
        self.track_compliance_history = deque(maxlen=1800)  # For compliance %
        self.critical_alerts_30min = 0
        self.zone_a_detections = 0
        self.zone_a_compliant = 0
        self.zone_b_detections = 0
        self.zone_b_compliant = 0

        self.lock = threading.Lock()

    def add_detection(self, track: Dict) -> None:
        """Add detection to history for stats."""
        with self.lock:
            zone = track.get("compliance", {}).get("zone", "UNKNOWN")
            compliant = track.get("compliance", {}).get("compliant", False)

            # Track zone compliance
            if zone == "A":
                self.zone_a_detections += 1
                if compliant:
                    self.zone_a_compliant += 1
            elif zone == "B":
                self.zone_b_detections += 1
                if compliant:
                    self.zone_b_compliant += 1

            # Track overall compliance for compliance %
            self.track_compliance_history.append(compliant)

    def add_alert(self, alert: Dict) -> None:
        """Add alert to history."""
        with self.lock:
            self.events_deque.append(alert)
            if alert.get("alert_level") == "CRITICAL":
                self.critical_alerts_30min += 1

    def get_stats(self) -> Dict[str, Any]:
        """Compute and return stats."""
        with self.lock:
            # Compliance %
            if len(self.track_compliance_history) > 0:
                compliant_count = sum(self.track_compliance_history)
                compliance_pct = (compliant_count / len(self.track_compliance_history)) * 100.0
            else:
                compliance_pct = 100.0

            # Zone-specific compliance
            zone_a_comp = (
                (self.zone_a_compliant / self.zone_a_detections * 100.0)
                if self.zone_a_detections > 0
                else 0.0
            )
            zone_b_comp = (
                (self.zone_b_compliant / self.zone_b_detections * 100.0)
                if self.zone_b_detections > 0
                else 0.0
            )

            return {
                "compliance_pct": round(compliance_pct, 1),
                "total_entered": self.people_entered,
                "people_inside": self.people_inside,
                "critical_alerts_30min": self.critical_alerts_30min,
                "zone_a_compliance": round(zone_a_comp, 1),
                "zone_b_compliance": round(zone_b_comp, 1),
                "last_alert_time": (
                    self.events_deque[-1].get("timestamp")
                    if len(self.events_deque) > 0
                    else None
                ),
                "pi_connected": self.pi_connected,
                "camera_ok": self.camera_ok,
            }

    def update_from_detection(self, data: Dict) -> None:
        """Update state from detection message."""
        with self.lock:
            if isinstance(data, list):
                for track in data:
                    self.add_detection(track)
            elif isinstance(data, dict):
                self.add_detection(data)

    def update_from_frame_meta(self, data: Dict) -> None:
        """Update state from frame meta message."""
        with self.lock:
            self.people_inside = data.get("people_inside", 0)
            self.people_entered = data.get("people_entered", 0)
            self.current_zone = data.get("zone", "UNKNOWN")
            frame_flags = data.get("frame_flags", {})
            self.low_visibility = frame_flags.get("low_visibility", False)
            self.glare = frame_flags.get("glare", False)

    def update_from_alert(self, data: Dict) -> None:
        """Update state from alert message."""
        with self.lock:
            self.add_alert(data)


state = DashboardState()

# ============================================================================
# ROSBRIDGE WEBSOCKET CLIENT
# ============================================================================

class RosbridgeClient:
    """Connects to rosbridge_server and subscribes to vision topics."""

    def __init__(self, rosbridge_url: str, socketio_instance):
        self.rosbridge_url = rosbridge_url
        self.socketio = socketio_instance
        self.ws = None
        self.connected = False
        self.thread = None
        self.running = True
        self.msg_id_counter = 0
        self.subscriptions = {}

    def start(self) -> None:
        """Start WebSocket connection in background thread."""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        """Main WebSocket connection loop with auto-reconnect."""
        while self.running:
            try:
                self._connect_and_subscribe()
            except Exception as e:
                logger.error(f"[ROSBRIDGE] Connection error: {e}")
                state.pi_connected = False
                self._emit_connection_status()

            # Reconnect after 3 seconds
            time.sleep(3)

    def _connect_and_subscribe(self) -> None:
        """Connect to rosbridge and subscribe to topics."""
        logger.info(f"[ROSBRIDGE] Connecting to {self.rosbridge_url}")

        self.ws = websocket.create_connection(self.rosbridge_url, timeout=5)
        self.connected = True
        state.pi_connected = True
        self._emit_connection_status()
        logger.info("[ROSBRIDGE] Connected")

        # Subscribe to topics
        self._subscribe("/vision/detections")
        self._subscribe("/vision/frame_meta")
        self._subscribe("/vision/alerts")

        # Read messages
        while self.running and self.connected:
            try:
                msg = self.ws.recv(timeout=30)
                self._on_message(msg)
            except websocket.WebSocketTimeoutException:
                logger.warning("[ROSBRIDGE] Timeout waiting for message")
                self.connected = False
                break
            except Exception as e:
                logger.error(f"[ROSBRIDGE] Receive error: {e}")
                self.connected = False
                break

        if self.ws:
            self.ws.close()

    def _subscribe(self, topic: str) -> None:
        """Subscribe to a ROS topic."""
        self.msg_id_counter += 1
        msg_id = self.msg_id_counter

        subscribe_msg = {
            "op": "subscribe",
            "id": msg_id,
            "topic": topic,
            "type": "std_msgs/String",
        }

        self.ws.send(json.dumps(subscribe_msg))
        self.subscriptions[msg_id] = topic
        logger.info(f"[ROSBRIDGE] Subscribed to {topic}")

    def _on_message(self, msg_str: str) -> None:
        """Handle incoming ROS message."""
        try:
            msg = json.loads(msg_str)

            # Skip non-message events
            if msg.get("op") != "publish":
                return

            topic = msg.get("topic", "")
            msg_data = msg.get("msg", {})

            # Extract JSON data from std_msgs/String
            if "data" in msg_data:
                try:
                    data = json.loads(msg_data["data"])
                except (json.JSONDecodeError, TypeError):
                    data = msg_data["data"]
            else:
                data = msg_data

            # Route to appropriate handler
            if topic == "/vision/detections":
                self._on_detections(data)
            elif topic == "/vision/frame_meta":
                self._on_frame_meta(data)
            elif topic == "/vision/alerts":
                self._on_alert(data)

        except Exception as e:
            logger.error(f"[ROSBRIDGE] Message parse error: {e}")

    def _on_detections(self, data: Any) -> None:
        """Handle detection message."""
        try:
            state.update_from_detection(data)
            state.camera_ok = True

            # Emit to connected clients
            socketio_msg = {
                "tracks": data if isinstance(data, list) else [data],
                "people_inside": state.people_inside,
                "zone": state.current_zone,
            }
            self.socketio.emit("detection_update", socketio_msg, broadcast=True)

        except Exception as e:
            logger.error(f"[DETECTIONS] Error: {e}")

    def _on_frame_meta(self, data: Any) -> None:
        """Handle frame metadata message."""
        try:
            state.update_from_frame_meta(data)

            socketio_msg = {
                "people_entered": state.people_entered,
                "people_inside": state.people_inside,
                "low_visibility": state.low_visibility,
                "zone": state.current_zone,
            }
            self.socketio.emit("frame_meta_update", socketio_msg, broadcast=True)

        except Exception as e:
            logger.error(f"[FRAME_META] Error: {e}")

    def _on_alert(self, data: Any) -> None:
        """Handle alert message."""
        try:
            state.update_from_alert(data)

            socketio_msg = {
                "alert_level": data.get("alert_level", "OK"),
                "missing_ppe": data.get("missing_ppe", []),
                "track_id": data.get("track_id", 0),
                "zone": data.get("zone", "UNKNOWN"),
                "depth_m": data.get("depth_m", 0.0),
                "timestamp": data.get("timestamp", time.time()),
            }
            self.socketio.emit("alert_event", socketio_msg, broadcast=True)

        except Exception as e:
            logger.error(f"[ALERTS] Error: {e}")

    def _emit_connection_status(self) -> None:
        """Emit connection status to clients."""
        status = {
            "pi_connected": state.pi_connected,
            "camera_ok": state.camera_ok,
        }
        self.socketio.emit("connection_status", status, broadcast=True)

    def stop(self) -> None:
        """Stop the client."""
        self.running = False
        if self.ws:
            self.ws.close()


# ============================================================================
# BACKGROUND STATS EMISSION
# ============================================================================

def emit_stats_loop() -> None:
    """Emit stats every 10 seconds."""
    while True:
        try:
            time.sleep(10)
            stats = state.get_stats()
            socketio.emit("stats_update", stats, broadcast=True, skip_sid=True)
        except Exception as e:
            logger.error(f"[STATS] Error: {e}")


# ============================================================================
# ROUTES & EVENTS
# ============================================================================

@app.route("/")
def index():
    """Serve dashboard HTML."""
    with open(os.path.join(os.path.dirname(__file__), "index.html"), "r") as f:
        return f.read()


@app.route("/video_feed")
def video_feed():
    """Proxy MJPEG stream from Pi."""
    try:
        response = requests.get(MJPEG_STREAM_URL, stream=True, timeout=5)
        response.raise_for_status()

        def generate():
            for chunk in response.iter_content(chunk_size=1024):
                yield chunk

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Access-Control-Allow-Origin": "*"},
        )

    except Exception as e:
        logger.warning(f"[MJPEG] Stream unavailable: {e}")
        # Return placeholder (camera offline)
        return Response(
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n",
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )


@socketio.on("connect")
def on_connect():
    """Handle client connection."""
    logger.info(f"[SOCKETIO] Client connected")
    emit("connection_status", state.get_stats())


@socketio.on("disconnect")
def on_disconnect():
    """Handle client disconnection."""
    logger.info(f"[SOCKETIO] Client disconnected")


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    # Start rosbridge client
    rosbridge = RosbridgeClient(ROSBRIDGE_URL, socketio)
    rosbridge.start()

    # Start stats emission thread
    stats_thread = threading.Thread(target=emit_stats_loop, daemon=True)
    stats_thread.start()

    # Start Flask app
    logger.info("[APP] Starting Flask server on http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
