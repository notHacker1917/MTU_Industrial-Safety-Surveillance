#!/bin/bash
# Complete Safety Rover setup script for Raspberry Pi 5 + Ubuntu 24.04
# Idempotent - safe to run multiple times
# Run on Pi: bash setup.sh

set -e

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PATH="$HOME/rover_venv"
ROS2_WS="$REPO_ROOT/ros2_ws"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo ""
echo "=========================================="
echo "Safety Rover Setup Script"
echo "Repository: $REPO_ROOT"
echo "Venv: $VENV_PATH"
echo "ROS2 Workspace: $ROS2_WS"
echo "=========================================="
echo ""

# ============================================================================
# 1. SYSTEM PACKAGES
# ============================================================================

log_info "Step 1: Updating system packages..."
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    cmake \
    libopencv-dev \
    2>&1 | grep -E "^(Setting|Processing|Already)" || true

log_info "✓ System packages installed"

# ============================================================================
# 2. ROS2 JAZZY INSTALLATION
# ============================================================================

if ! command -v ros2 &> /dev/null; then
    log_info "Step 2: Installing ROS2 Jazzy..."
    
    # Add ROS2 GPG key
    sudo curl -sSL https://repo.ros2.org/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg 2>/dev/null || \
        log_warn "Could not add ROS2 GPG key (may already exist)"
    
    # Add ROS2 repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) main" | \
        sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    
    sudo apt-get update
    sudo apt-get install -y \
        ros-jazzy-ros-core \
        ros-jazzy-slam-toolbox \
        ros-jazzy-nav2-bringup \
        ros-jazzy-robot-localization \
        ros-jazzy-rosbridge-server \
        ros-jazzy-rplidar-ros \
        python3-colcon-common-extensions \
        2>&1 | grep -E "^(Setting|Processing|Already)" || true
    
    log_info "✓ ROS2 Jazzy installed"
else
    log_info "ROS2 already installed"
fi

# ============================================================================
# 3. PYTHON VIRTUAL ENVIRONMENT
# ============================================================================

if [ ! -d "$VENV_PATH" ]; then
    log_info "Step 3: Creating Python virtual environment at $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
    log_info "✓ Virtual environment created"
else
    log_info "Virtual environment already exists"
fi

source "$VENV_PATH/bin/activate"
log_info "✓ Virtual environment activated"

# ============================================================================
# 4. PYTHON DEPENDENCIES
# ============================================================================

log_info "Step 4: Installing Python dependencies..."
pip install --quiet --upgrade pip setuptools wheel

# Core dependencies
pip install --quiet \
    depthai \
    opencv-python \
    opencv-contrib-python \
    tflite-runtime \
    scipy \
    numpy \
    PyYAML \
    rclpy \
    bluepy \
    websocket-client \
    requests \
    flask \
    flask-socketio \
    flask-cors \
    eventlet \
    python-socketio \
    pytest \
    2>&1 | tail -n 5

log_info "✓ Python dependencies installed"

# ============================================================================
# 5. DepthAI udev RULES (for USB access without sudo)
# ============================================================================

log_info "Step 5: Installing DepthAI udev rules..."
if [ ! -f /etc/udev/rules.d/80-movidius.rules ]; then
    echo 'SUBSYSTEM=="usb",ATTRS{idVendor}=="03e7",MODE="0666"' | \
        sudo tee /etc/udev/rules.d/80-movidius.rules > /dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    log_info "✓ udev rules installed"
else
    log_info "udev rules already exist"
fi

# ============================================================================
# 6. ROS2 WORKSPACE BUILD
# ============================================================================

if [ -d "$ROS2_WS/src" ]; then
    log_info "Step 6: Building ROS2 workspace..."
    cd "$ROS2_WS"
    
    # Source ROS2
    source /opt/ros/jazzy/setup.bash
    
    # Clean if requested
    if [ "$1" == "--clean" ]; then
        log_warn "Cleaning workspace (--clean flag)"
        rm -rf build install log
    fi
    
    # Build
    colcon build --packages-select vision_pkg navigation_pkg 2>&1 | tail -n 10
    
    log_info "✓ ROS2 workspace built"
else
    log_warn "ROS2 workspace not found at $ROS2_WS/src"
fi

# ============================================================================
# 7. SHELL CONFIGURATION
# ============================================================================

log_info "Step 7: Updating shell configuration..."

# Add ROS2 setup to .bashrc
if ! grep -q "source /opt/ros/jazzy/setup.bash" "$HOME/.bashrc"; then
    echo "source /opt/ros/jazzy/setup.bash" >> "$HOME/.bashrc"
    log_info "✓ Added ROS2 setup to ~/.bashrc"
fi

# Add workspace setup to .bashrc
if ! grep -q "source $ROS2_WS/install/setup.bash" "$HOME/.bashrc"; then
    echo "source $ROS2_WS/install/setup.bash" >> "$HOME/.bashrc"
    log_info "✓ Added workspace setup to ~/.bashrc"
fi

# Add venv activation to .bashrc (optional but helpful)
if [ ! -f "$HOME/.bashrc.bak.safety_rover" ]; then
    log_info "To activate venv automatically, add this to ~/.bashrc:"
    log_info "  source $VENV_PATH/bin/activate"
fi

# ============================================================================
# 8. DIRECTORIES & PERMISSIONS
# ============================================================================

log_info "Step 8: Setting up directories..."
mkdir -p "$REPO_ROOT/logs"
mkdir -p "$REPO_ROOT/models"
chmod +x "$REPO_ROOT/setup.sh"
chmod +x "$REPO_ROOT/launch_all.sh"
chmod +x "$REPO_ROOT/models/download_models.sh"
log_info "✓ Directories ready"

# ============================================================================
# 9. MODEL SETUP (OPTIONAL)
# ============================================================================

if [ ! -f "$REPO_ROOT/models/yolov26n_320_320.blob" ]; then
    log_warn "Step 9: Models not found. To download:"
    log_warn "  export GDRIVE_ID=<folder-id>"
    log_warn "  bash $REPO_ROOT/models/download_models.sh"
else
    log_info "✓ Models found"
fi

# ============================================================================
# 10. VERIFICATION
# ============================================================================

log_info "Step 10: Verifying setup..."

# Check OAK-D connectivity (only if on actual Pi)
if [ -x "$(command -v lsusb)" ]; then
    if lsusb | grep -q "03e7"; then
        log_info "✓ OAK-D camera detected"
    else
        log_warn "OAK-D camera not detected (may be disconnected)"
    fi
fi

# Check ROS2
if [ -x "$(command -v ros2)" ]; then
    log_info "✓ ROS2 installed: $(ros2 --version | head -1)"
else
    log_error "ROS2 not found - setup may have failed"
    exit 1
fi

# Check Python packages
python3 -c "import depthai; import cv2; import yaml; import rclpy" 2>/dev/null && \
    log_info "✓ All Python packages available" || \
    log_error "Some Python packages missing"

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Download models (if not already present):"
echo "   export GDRIVE_ID=<Google-Drive-folder-ID>"
echo "   bash $REPO_ROOT/models/download_models.sh"
echo ""
echo "2. Source your environment:"
echo "   source ~/.bashrc"
echo "   source $VENV_PATH/bin/activate"
echo ""
echo "3. Verify configuration:"
echo "   cd $REPO_ROOT"
echo "   python3 config/config_loader.py"
echo ""
echo "4. Launch all nodes:"
echo "   bash $REPO_ROOT/launch_all.sh"
echo ""
echo "5. (On laptop) Connect to dashboard:"
echo "   bash $REPO_ROOT/dashboard/start_dashboard.sh 192.168.1.100"
echo "   # Open http://localhost:5000 in browser"
echo ""
echo "Docs: $REPO_ROOT/README.md"
echo "Config: $REPO_ROOT/config/rover_params.yaml"
echo ""
