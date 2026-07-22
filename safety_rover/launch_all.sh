#!/bin/bash
# Master launch script for Safety Rover
# Launches all ROS2 nodes in correct dependency order with logging
# Usage: bash launch_all.sh [--no-nav] [--clean-logs]

set -e

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROS2_WS="$REPO_ROOT/ros2_ws"
LOGS_DIR="$REPO_ROOT/logs"
LAUNCH_PID_FILE="$LOGS_DIR/.launch.pids"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_node() { echo -e "${BLUE}[NODE]${NC} $1"; }

# Parse arguments
LAUNCH_NAV=true
CLEAN_LOGS=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-nav)
            LAUNCH_NAV=false
            log_warn "Navigation disabled (--no-nav)"
            shift
            ;;
        --clean-logs)
            CLEAN_LOGS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# SETUP
# ============================================================================

echo ""
echo "=========================================="
echo "Safety Rover Master Launch"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p "$LOGS_DIR"

# Clean logs if requested
if [ "$CLEAN_LOGS" = true ]; then
    log_warn "Cleaning logs..."
    rm -f "$LOGS_DIR"/*.log
fi

# Initialize PID file
> "$LAUNCH_PID_FILE"

# Cleanup function (on Ctrl+C)
cleanup() {
    log_warn "Shutting down all nodes..."
    
    # Kill all ROS2 processes
    if [ -f "$LAUNCH_PID_FILE" ]; then
        while IFS= read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
                sleep 0.5
            fi
        done < "$LAUNCH_PID_FILE"
    fi
    
    # Kill any remaining ros2 processes
    pkill -f "ros2" || true
    
    echo ""
    log_info "✓ All nodes shut down"
    exit 0
}

trap cleanup SIGINT SIGTERM

# ============================================================================
# PREREQUISITES
# ============================================================================

log_info "Checking prerequisites..."

# Check OAK-D connectivity
if [ -x "$(command -v lsusb)" ]; then
    if ! lsusb | grep -q "03e7"; then
        log_error "OAK-D camera not detected!"
        echo "  Troubleshooting:"
        echo "    1. Check USB connection"
        echo "    2. Run: lsusb | grep -i oak"
        echo "    3. Check udev rules: cat /etc/udev/rules.d/80-movidius.rules"
        exit 1
    fi
    log_info "✓ OAK-D camera detected"
else
    log_warn "lsusb not available (might be in Docker/WSL)"
fi

# Check RPLidar
if [ -x "$(command -v ls)" ]; then
    if ls /dev/ttyUSB* 2>/dev/null | head -1 > /dev/null; then
        log_info "✓ Serial device detected (RPLidar/HCSR04)"
    else
        log_warn "No /dev/ttyUSB* found (RPLidar may not be connected)"
    fi
fi

# Check models exist
if [ ! -f "$REPO_ROOT/models/yolov26n_320_320.blob" ]; then
    log_error "Model not found: $REPO_ROOT/models/yolov26n_320_320.blob"
    echo "  Download with:"
    echo "    export GDRIVE_ID=<folder-id>"
    echo "    bash $REPO_ROOT/models/download_models.sh"
    exit 1
fi

if [ ! -f "$REPO_ROOT/models/ppe_classifier_model.tflite" ]; then
    log_error "Model not found: $REPO_ROOT/models/ppe_classifier_model.tflite"
    exit 1
fi

log_info "✓ All prerequisites met"
echo ""

# ============================================================================
# SOURCE ROS2 & WORKSPACE
# ============================================================================

log_info "Sourcing ROS2 environment..."
if [ ! -f "/opt/ros/jazzy/setup.bash" ]; then
    log_error "ROS2 not installed at /opt/ros/jazzy"
    exit 1
fi
source /opt/ros/jazzy/setup.bash

if [ -f "$ROS2_WS/install/setup.bash" ]; then
    source "$ROS2_WS/install/setup.bash"
    log_info "✓ ROS2 workspace sourced"
else
    log_warn "Workspace not built yet. Build with: cd $ROS2_WS && colcon build"
fi

echo ""

# ============================================================================
# LAUNCH NODES (In Dependency Order)
# ============================================================================

log_info "Launching nodes..."
echo ""

# 1. NAVIGATION (depends on: nothing)
if [ "$LAUNCH_NAV" = true ]; then
    if ros2 pkg list | grep -q "navigation_pkg"; then
        log_node "Launching Navigation Stack..."
        ros2 launch navigation_pkg navigation_launch.py \
            > "$LOGS_DIR/nav.log" 2>&1 &
        NAV_PID=$!
        echo "$NAV_PID" >> "$LAUNCH_PID_FILE"
        log_node "  PID: $NAV_PID"
        sleep 5  # Wait for Nav2 to initialize
    else
        log_warn "navigation_pkg not found (Person B may not have pushed yet)"
    fi
else
    log_warn "Navigation stack skipped (--no-nav)"
fi

# 2. VISION (depends on: OAK-D camera, models)
log_node "Launching Vision Stack..."
ros2 launch vision_pkg vision_launch.py \
    blob_path:="$REPO_ROOT/models/yolov26n_320_320.blob" \
    tflite_model_path:="$REPO_ROOT/models/ppe_classifier_model.tflite" \
    > "$LOGS_DIR/vision.log" 2>&1 &
VISION_PID=$!
echo "$VISION_PID" >> "$LAUNCH_PID_FILE"
log_node "  PID: $VISION_PID"
sleep 3

# 3. RosBridge (depends on: ROS2 master)
log_node "Launching RosBridge..."
ros2 run rosbridge_server rosbridge_websocket \
    > "$LOGS_DIR/rosbridge.log" 2>&1 &
ROSBRIDGE_PID=$!
echo "$ROSBRIDGE_PID" >> "$LAUNCH_PID_FILE"
log_node "  PID: $ROSBRIDGE_PID"
sleep 2

echo ""
log_info "✓ All nodes launched"
echo ""
echo "Node Status:"
echo "  Navigation:    $([ "$LAUNCH_NAV" = true ] && echo "✓ Running (PID $NAV_PID)" || echo "⊘ Skipped")"
echo "  Vision:        ✓ Running (PID $VISION_PID)"
echo "  RosBridge:     ✓ Running (PID $ROSBRIDGE_PID)"
echo ""
echo "Logs:"
echo "  Vision:        $LOGS_DIR/vision.log"
echo "  Navigation:    $LOGS_DIR/nav.log"
echo "  RosBridge:     $LOGS_DIR/rosbridge.log"
echo ""
echo "Monitor:"
echo "  Tail vision:     tail -f $LOGS_DIR/vision.log"
echo "  Tail all:        tail -f $LOGS_DIR/*.log"
echo "  ROS2 topics:     ros2 topic list"
echo "  ROS2 echo:       ros2 topic echo /vision/detections"
echo ""
echo "Dashboard:"
echo "  Start:   bash $REPO_ROOT/dashboard/start_dashboard.sh"
echo "  URL:     http://localhost:5000"
echo ""
echo "Press Ctrl+C to shut down all nodes"
echo "=========================================="
echo ""

# ============================================================================
# MONITORING
# ============================================================================

# Tail vision logs to stdout for monitoring
tail -f "$LOGS_DIR/vision.log" &
TAIL_PID=$!
echo "$TAIL_PID" >> "$LAUNCH_PID_FILE"

# Keep script alive
wait
