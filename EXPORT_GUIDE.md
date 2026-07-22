# YOLO11n Export Pipeline — A1 + A2 Complete Guide

**Goal:** Convert YOLO11n → ONNX → Blob in one command (laptop, no Pi needed yet)

---

## Quick Start

```bash
# On your laptop (Windows/Mac/Linux)
python3 export_to_blob.py
```

**That's it.** The script handles everything: A1 (YOLO→ONNX) + A2 (ONNX→Blob).

---

## Prerequisites (Auto-Checked)

The script will check for and install these if missing:

- **ultralytics** ≥8.2.0 — YOLO framework (auto-downloads yolo11n.pt)
- **onnx** ≥1.14.0 — ONNX model format
- **onnxsim** ≥0.4.0 — ONNX graph simplification
- **blobconverter** ≥0.6.7 — Automated blob conversion

If any are missing, the script prints the exact `pip install` command. Just run it and re-run `export_to_blob.py`.

---

## What the Script Does

### A1: YOLO → ONNX Export
1. ✅ Check ultralytics installed
2. ✅ Download yolo11n.pt (~6 MB, one-time)
3. ✅ Export with YOLO settings:
   - Input size: 320×320×3 (fixed for VPU)
   - Format: ONNX opset=12 (OpenVINO compatible)
   - Simplify: True (reduce graph complexity)
   - Dynamic: False (static shapes required)
4. ✅ Validate ONNX model
5. ✅ Extract metadata (input/output shapes, tensor names)
6. ✅ Save to: **yolo11n.onnx**

### A2: ONNX → Blob Conversion
1. ✅ Attempt automated conversion via blobconverter API
   - Contacts OpenVINO backend
   - Converts to MyriadX format
   - Downloads .blob file
   - Takes 30–120 seconds
2. ✅ If automated fails → Print manual fallback instructions
   - Web tool: http://tools.luxonis.com (2–5 min)
   - You upload ONNX, download blob
3. ✅ Validate blob file size (>2 MB = valid)
4. ✅ Save to: **models/yolo11n.blob**

### Supporting Files Generated
- **onnx_info.txt** — Metadata for A3 (tensor parsing)
- **requirements_export.txt** — Reproducible deps for team
- **copy_to_pi.sh** — One-command deployment to Pi

---

## Execution Flow

```
export_to_blob.py
├─ Step 0: Check dependencies
│  ├─ ultralytics? ✓
│  ├─ onnx? ✓
│  ├─ onnxsim? ✓
│  └─ (auto-install if missing)
│
├─ Step A1: Export YOLO11n to ONNX
│  ├─ Load yolo11n.pt (auto-download)
│  ├─ Export to ONNX (opset=12, imgsz=320, simplify=True, dynamic=False)
│  ├─ Validate ONNX model
│  ├─ Extract input/output shapes
│  └─ Save: yolo11n.onnx + onnx_info.txt
│
├─ Step A2: Convert ONNX to Blob
│  ├─ Check blobconverter
│  ├─ Try automated conversion (API call to OpenVINO backend)
│  │  └─ If success: models/yolo11n.blob ✓
│  │  └─ If fail: Print manual instructions
│  └─ Validate blob size
│
├─ Step 3: Generate deployment files
│  ├─ requirements_export.txt
│  └─ copy_to_pi.sh
│
└─ Step 4: Print summary & next steps
   └─ Ready for Pi!
```

---

## Expected Output

### ✅ Success Case (Automated Conversion Works)

```
=============== FINAL CHECKLIST — A1 + A2 ===============

FILES GENERATED:
  [✓] yolo11n.onnx (6.50 MB)
  [✓] onnx_info.txt
  [✓] requirements_export.txt
  [✓] copy_to_pi.sh
  [✓] models/yolo11n.blob (9.23 MB)

STATUS: ✓ READY FOR PI

NEXT STEPS:
  1. Verify blob exists: ls -lh models/
  2. Share models/ with team (upload to shared storage)
  3. Deploy to Pi: bash copy_to_pi.sh 192.168.1.100
  4. On Pi verify: ls -lh ~/safety_rover/models/
  5. Update oak_pipeline.py with: blob_path='models/yolo11n.blob'

OUTPUT METADATA (saved in onnx_info.txt):
  Input shape: {'images': [1, 3, 320, 320]}
  Output names: ['output0', 'output1']
  Output shapes: {...}

Share onnx_info.txt with Person C for A3 tensor parsing
```

### ⚠️ Manual Conversion Case (Automated Fails)

```
AUTOMATED CONVERSION FAILED — MANUAL FALLBACK

MANUAL CONVERSION VIA WEB TOOL (takes ~2-5 minutes):

  1. Open browser: http://tools.luxonis.com
  2. Upload file: yolo11n.onnx
  3. Select conversion options:
     • Target device: MyriadX
     • Data type: FP16
     • Shaves: 6
     • OpenVINO version: 2022.1
  4. Click 'Convert' button (wait ~2-5 minutes)
  5. Download the .blob file
  6. Save to: ./models/yolo11n.blob
```

---

## Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'ultralytics'`

**Solution:** The script will print the exact pip command. Just run:
```bash
pip install ultralytics>=8.2.0 onnx>=1.14.0 onnxsim>=0.4.0 blobconverter>=0.6.7
python3 export_to_blob.py
```

### Problem: Script hangs at "Converting (this may take 30-120 seconds)..."

**This is normal!** The script is uploading your ONNX to the OpenVINO backend and waiting for conversion. This can take 1–3 minutes. Don't interrupt.

If it times out (>5 min):
1. Press Ctrl+C
2. Use manual conversion: http://tools.luxonis.com
3. Download blob and save to `models/yolo11n.blob`

### Problem: Blob file is missing or too small (<2 MB)

**This means automated conversion failed.** The script will print manual instructions. Use the web tool: http://tools.luxonis.com

### Problem: `rsync: command not found` when running `copy_to_pi.sh`

This script requires **rsync** (installed by default on macOS/Linux). On Windows, you need:
- **Option 1:** Use Windows Subsystem for Linux (WSL2)
- **Option 2:** Use Git Bash (includes rsync)
- **Option 3:** Manually copy blob file via WinSCP or `scp`

---

## After Export — Deploy to Pi

### 1. Verify blob exists locally
```bash
ls -lh models/
# Output: yolo11n.blob (should be 8–12 MB)
```

### 2. Deploy to Pi (one command)
```bash
bash copy_to_pi.sh 192.168.1.100
# or with username:
bash copy_to_pi.sh ubuntu@192.168.1.100
```

### 3. Verify on Pi
```bash
ssh ubuntu@192.168.1.100 'ls -lh ~/safety_rover/models/'
# Output: yolo11n.blob (same size as local)
```

### 4. Update oak_pipeline.py
Open `oak_pipeline.py` and set:
```python
blob_path = "models/yolo11n.blob"  # Or full path: /home/ubuntu/safety_rover/models/yolo11n.blob
```

---

## File Descriptions

| File | Purpose | Generated By |
|------|---------|--------------|
| `export_to_blob.py` | Main script (A1 + A2) | You (manual) |
| `yolo11n.onnx` | YOLO exported to ONNX format | Script (A1) |
| `models/yolo11n.blob` | Blob compiled for MyriadX VPU | Script (A2) |
| `onnx_info.txt` | Input/output shapes + tensor names | Script (A1) |
| `requirements_export.txt` | Python dependencies (for team) | Script (A3) |
| `copy_to_pi.sh` | Deployment script | Script (A3) |

---

## What Gets Shared with Team

After running the script, share these files:

1. **onnx_info.txt** — Metadata needed for A3 tensor parsing
2. **models/yolo11n.blob** — The compiled model
3. **requirements_export.txt** — If team wants to reproduce locally

Don't share:
- `yolo11n.onnx` (not needed on Pi, takes up space)
- `yolo11n.pt` (auto-downloaded when needed)

---

## Next Steps (After A1 + A2 Complete)

### For Person A (Vision Lead):
- [ ] Update oak_pipeline.py to use blob path
- [ ] Test blob inference locally with mock data
- [ ] Prepare for A3 tensor parsing (use onnx_info.txt)

### For Person B (Nav Lead):
- [ ] No action needed until Person C deploys to Pi

### For Person C (Integration Lead):
- [ ] Deploy blob to Pi: `bash copy_to_pi.sh <pi-ip>`
- [ ] Verify blob present: `ssh pi@<ip> 'ls -lh ~/safety_rover/models/'`
- [ ] Update launch_all.sh with blob_path if needed

---

## Command Cheatsheet

```bash
# Export locally
python3 export_to_blob.py

# Check what was generated
ls -lh models/yolo*.blob
cat onnx_info.txt

# Deploy to Pi
bash copy_to_pi.sh 192.168.1.100

# Verify on Pi
ssh ubuntu@192.168.1.100 'ls -lh ~/safety_rover/models/'

# If manual conversion, check blob size
python3 -c "from pathlib import Path; \
print(f'Blob size: {Path(\"models/yolo11n.blob\").stat().st_size / (1024**2):.2f} MB')"
```

---

## Performance Notes

- **A1 (Export):** 10–30 seconds
- **A2 (Automated conversion):** 30–120 seconds (depends on OpenVINO backend load)
- **A2 (Manual conversion):** 2–5 minutes (you wait in browser)
- **Total time:** 1–3 minutes (automated) or 2–5 minutes (manual)

---

## Reference

- **YOLO11n specs:** 6.5 MB (model) → 6.5 MB (ONNX) → 9–12 MB (blob)
- **Input:** 320×320×3 BGR
- **Output:** YOLO detection heads (format depends on export)
- **VPU:** Myriad X (2.0 TOPS, 6 shaves optimal)
- **Latency:** ~33 ms @ 30 FPS

---

**Last Updated:** 2026-06-16 | **Status:** Ready to Export ✨
