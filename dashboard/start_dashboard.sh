#!/bin/bash
# Dashboard Startup Script
# Sets configuration and starts Flask server

# Get Pi IP from argument or use default
ROVER_PI_IP="${1:-192.168.1.100}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Export configuration
export ROVER_PI_IP="$ROVER_PI_IP"
export ROVER_CAM_PORT="${ROVER_CAM_PORT:-5800}"
export ROVER_ROSBRIDGE_PORT="${ROVER_ROSBRIDGE_PORT:-9090}"

echo "================================================"
echo "   PPE Compliance Dashboard"
echo "================================================"
echo ""
echo "Configuration:"
echo "  Pi IP: $ROVER_PI_IP"
echo "  Camera Port: $ROVER_CAM_PORT"
echo "  RosBridge Port: $ROVER_ROSBRIDGE_PORT"
echo ""

# Activate venv if exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    echo "Activating virtual environment..."
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Install requirements silently
echo "Installing dependencies..."
pip install -q -r "$SCRIPT_DIR/requirements.txt" 2>/dev/null

# Run Flask app
echo ""
echo "Starting Flask server..."
echo ""
echo "================================================"
echo "  Dashboard ready → http://localhost:5000"
echo "================================================"
echo ""

cd "$SCRIPT_DIR"
python app.py
