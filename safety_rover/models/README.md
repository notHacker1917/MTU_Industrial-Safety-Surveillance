# Models Directory

This directory contains machine learning models for Safety Rover inference.

## ⚠️ IMPORTANT: Models Not in Git

Models are **NOT committed to Git** because they're too large (>100MB). Instead:

1. **Host externally** on a shared service (Google Drive, Dropbox, etc.)
2. **Download before first run** using the provided script
3. **Verify checksums** to ensure integrity

## Model Files Required

| Model | Purpose | Size | Format | Source |
|-------|---------|------|--------|--------|
| `yolov26n_320_320.blob` | Person detection on OAK-D VPU | ~10MB | OAK Binary (.blob) | YOLOv2.6 compiled for MyriadX |
| `ppe_classifier_model.tflite` | PPE classification (suit/shield/gloves) | ~3MB | TFLite INT8 | TensorFlow Lite quantized |

## Download Instructions

### Option 1: Automated Download (Recommended)

```bash
# Set Google Drive folder ID (shared by team)
export GDRIVE_ID="<folder-id-from-Drive>"

# Run download script
bash download_models.sh
```

The script will:
- Check for gdown (Google Drive CLI tool)
- Download all models from shared folder
- Verify files exist and report sizes

### Option 2: Manual Download

1. Ask Person C or team lead for shared Google Drive link
2. Download files to this directory:
   ```bash
   # Example using gdown
   gdown <file-id> -O .
   ```
3. Verify:
   ```bash
   ls -lh *.blob *.tflite
   ```

### Option 3: Use Local Models

If models are already on Pi:
```bash
# Copy from external drive
cp /mnt/usb/models/* .

# Or build/download separately
python3 -c "import depthai; print(depthai.Pipeline().getCameraProperties())"
```

## Model Specifications

### yolov26n_320_320.blob

**Detection Model** (Person only)
- **Input**: RGB image 320×320
- **Output**: Detections with bbox, confidence, class_id
- **Confidence Threshold**: 0.45 (configurable in `config/rover_params.yaml`)
- **NMS IoU Threshold**: 0.5
- **Compilation Target**: OAK-D MyriadX VPU
- **Performance**: 30 FPS on VPU with <100ms latency

**Usage in Code**:
```python
from oak_pipeline import OakDPipeline

pipeline = OakDPipeline(blob_path="models/yolov26n_320_320.blob")
frame, detections = pipeline.get_frame()
```

### ppe_classifier_model.tflite

**Classification Model** (Multi-label)
- **Input**: Cropped RGB image 224×224 (from detection bbox)
- **Output**: 3 sigmoid outputs [suit_score, shield_score, gloves_score]
- **Quantization**: INT8 (post-training quantization)
- **Confidence Thresholds** (configurable):
  - Suit: 0.65
  - Shield: 0.60
  - Gloves: 0.65
- **Performance**: <50ms per inference on Pi CPU

**Usage in Code**:
```python
from ppe_classifier import PPEClassifier

classifier = PPEClassifier(
    model_path="models/ppe_classifier_model.tflite",
    conf_threshold=0.65
)
result = classifier.classify(cropped_bgr_image)
# Returns: {"suit": True/False, "shield": True/False, "gloves": True/False, ...}
```

## Validation & Checksums

After downloading, verify integrity:

```bash
# List downloaded files
ls -lh

# Expected sizes (approximate):
# yolov26n_320_320.blob: 9-12 MB
# ppe_classifier_model.tflite: 2-4 MB

# Check file type
file *.blob *.tflite
```

If checksums are available (ask team lead):
```bash
sha256sum *.blob *.tflite
# Compare with provided checksums
```

## Troubleshooting

### "Module not found: yolov26n_320_320.blob"

```bash
# Check current directory
pwd  # Should be repo root or ros2_ws/src/vision_pkg

# Check if models exist
ls -la models/

# If missing, download
export GDRIVE_ID="<ID>"
bash models/download_models.sh
```

### "gdown: command not found"

```bash
pip install gdown
bash models/download_models.sh
```

### Download fails or times out

- Check internet connection: `ping google.com`
- Verify folder is shared: Ask team member for new link
- Try manual download from Google Drive web interface
- Check if space available: `df -h` (need 50MB free)

### Model produces wrong output

- Verify model format: `yolov26n_320_320.blob` must be OAK-D compiled
- Check input size: Expect 320×320 RGB for YOLO, 224×224 RGB for PPE
- Check thresholds: `config/rover_params.yaml` confidence_threshold
- Update config if model differs from default (notify Person A)

## Model Training (Future)

To use custom models:

1. **Train/fine-tune** on your dataset
2. **Compile for OAK-D**: Use OAK converter tool
3. **Quantize TFLite**: Use TensorFlow's quantization API
4. **Replace files** in this directory
5. **Update config** if thresholds change
6. **Document** in this README (Person A responsible)

## References

- [OAK-D SDK](https://docs.luxonis.com/projects/api/en/latest/)
- [MyriadX Compilation](https://docs.luxonis.com/projects/api/en/latest/references/supported_models/)
- [TensorFlow Lite Conversion](https://www.tensorflow.org/lite/convert)
- [YOLO OAK Integration](https://github.com/luxonis/depthai-gstreamer)

---

**Owner**: Person A (Vision)  
**Last Updated**: Hackathon 2026  
**Status**: Models not yet uploaded to Drive (set GDRIVE_ID before first run)
