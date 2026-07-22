#!/bin/bash
# Deploy models/ folder to Raspberry Pi
# Usage: bash copy_to_pi.sh 192.168.1.100
#   or:  bash copy_to_pi.sh ubuntu@192.168.1.100

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <pi-ip-or-user@ip>"
    echo "Examples:"
    echo "  bash $0 192.168.1.100"
    echo "  bash $0 ubuntu@192.168.1.100"
    exit 1
fi

PI_HOST=$1

# Verify models directory exists
if [ ! -d "models" ]; then
    echo "ERROR: models/ directory not found in current directory"
    exit 1
fi

# Count .blob files
BLOB_COUNT=$(find models -name "*.blob" -type f | wc -l)
if [ $BLOB_COUNT -eq 0 ]; then
    echo "ERROR: No .blob files found in models/"
    echo "Please run: python3 export_to_blob.py"
    exit 1
fi

echo "=========================================="
echo "Deploying models/ to $PI_HOST"
echo "=========================================="
echo ""

# Deploy only .blob and metadata files
rsync -avz --progress \
    --include="*.blob" \
    --include="*.md" \
    --include="onnx_info.txt" \
    --exclude="*" \
    models/ \
    $PI_HOST:~/safety_rover/models/ 2>/dev/null || {
    echo "ERROR: rsync failed (check IP/SSH access)"
    exit 1
}

echo ""
echo "=========================================="
echo "✓ Models deployed successfully"
echo "=========================================="
echo ""
echo "Verify on Pi with:"
echo "  ssh $PI_HOST 'ls -lh ~/safety_rover/models/'"
