#!/bin/bash
# Download DepthAI models for Safety Rover
# Set GDRIVE_ID environment variable to your shared Google Drive folder ID
# Models are too large for Git LFS, so hosted externally

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS_DIR="$SCRIPT_DIR"

echo "=========================================="
echo "Downloading Safety Rover Models"
echo "=========================================="
echo ""

# Check if GDRIVE_ID is set
if [ -z "$GDRIVE_ID" ]; then
    echo "ERROR: GDRIVE_ID environment variable not set"
    echo ""
    echo "To download models, set:"
    echo "  export GDRIVE_ID=<Google Drive folder ID>"
    echo ""
    echo "Then run: bash $0"
    exit 1
fi

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown (Google Drive downloader)..."
    pip install gdown
fi

echo "Downloading from Google Drive folder: $GDRIVE_ID"
echo ""

# Download models (example file IDs - replace with actual IDs)
echo "1. Downloading YOLO26n.blob (OAK-D compiled model)..."
gdown --folder-id "$GDRIVE_ID" -O "$MODELS_DIR" --no-cookies --quiet 2>/dev/null || \
    echo "  ⚠ Download failed. Check GDRIVE_ID and verify files are shared."

echo ""
echo "2. Verifying model files..."

# Check required models
REQUIRED_FILES=(
    "yolov26n_320_320.blob"
    "ppe_classifier_model.tflite"
)

MISSING=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$MODELS_DIR/$file" ]; then
        SIZE=$(du -h "$MODELS_DIR/$file" | cut -f1)
        echo "  ✓ $file ($SIZE)"
    else
        echo "  ✗ MISSING: $file"
        MISSING=$((MISSING + 1))
    fi
done

echo ""
if [ $MISSING -eq 0 ]; then
    echo "✓ All models downloaded successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Verify models are compatible:"
    echo "     - yolov26n_320_320.blob must be compiled for OAK-D MyriadX"
    echo "     - ppe_classifier_model.tflite must be INT8 quantized"
    echo "  2. Update config/rover_params.yaml if thresholds differ"
    echo "  3. Run: bash launch_all.sh"
else
    echo "✗ $MISSING model(s) missing"
    exit 1
fi
