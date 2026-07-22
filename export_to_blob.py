#!/usr/bin/env python3
"""
A1 + A2: YOLO11n Export to ONNX → MyriadX Blob Pipeline
Complete workflow: yolo11n.pt → yolo11n.onnx → yolo11n.blob (ready for OAK-D)

Runs entirely on laptop (Windows/Linux/Mac) — all prerequisites auto-checked.
Author: Computer Vision Expert | Target: Raspberry Pi 5 + OAK-D MyriadX
"""

import subprocess
import sys
import os
import json
from pathlib import Path


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 75)
    print(f"{title.center(75)}")
    print("=" * 75)


def check_and_install_dependencies():
    """Check required packages; print exact pip command if missing."""
    print_section("STEP 0: DEPENDENCY CHECK")
    
    required = {
        'ultralytics': '>=8.2.0',
        'onnx': '>=1.14.0',
        'onnxsim': '>=0.4.0',
    }
    
    missing = []
    for pkg, version in required.items():
        try:
            __import__(pkg)
            print(f"  ✓ {pkg} found")
        except ImportError:
            missing.append(f"{pkg}{version}")
            print(f"  ✗ {pkg} NOT FOUND")
    
    if missing:
        print("\n" + "!" * 75)
        print("MISSING DEPENDENCIES — RUN THIS COMMAND:")
        print("!" * 75)
        cmd = f"pip install {' '.join(missing)}"
        print(f"\n  {cmd}\n")
        print("Then re-run this script.\n")
        sys.exit(1)
    
    print("\n✓ All required packages available")


def export_yolo_to_onnx():
    """
    A1: Export YOLO11n to ONNX with MyriadX-compatible settings.
    Returns: (onnx_path, metadata_dict)
    """
    print_section("A1: EXPORT YOLO11N → ONNX")
    
    try:
        from ultralytics import YOLO
        import onnx
    except ImportError as e:
        print(f"ERROR during import: {e}")
        sys.exit(1)
    
    print("\n1. Loading YOLO11n model (auto-downloads ~6MB)...")
    try:
        model = YOLO("yolo11n.pt")
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ ERROR: Failed to load model: {e}")
        sys.exit(1)
    
    print("\n2. Exporting to ONNX (imgsz=320, opset=12, dynamic=False)...")
    export_params = {
        'format': 'onnx',
        'imgsz': 320,           # Fixed for VPU
        'opset': 12,            # OpenVINO requirement
        'simplify': True,       # Reduce graph complexity
        'dynamic': False,       # Static shape required for MyriadX
    }
    
    try:
        results = model.export(**export_params)
        onnx_path = Path(results)
        
        if not onnx_path.exists():
            print(f"   ✗ ERROR: Export failed, file not found: {onnx_path}")
            sys.exit(1)
        
        onnx_size_mb = onnx_path.stat().st_size / (1024**2)
        print(f"   ✓ ONNX exported: {onnx_path.name} ({onnx_size_mb:.2f} MB)")
    except Exception as e:
        print(f"   ✗ ERROR during export: {e}")
        sys.exit(1)
    
    print("\n3. Validating ONNX model structure...")
    try:
        model_onnx = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_onnx)
        print("   ✓ ONNX model validation passed")
    except Exception as e:
        print(f"   ✗ ERROR: ONNX validation failed: {e}")
        sys.exit(1)
    
    print("\n4. Extracting input/output metadata...")
    graph = model_onnx.graph
    
    # Extract input shapes
    input_info = {}
    for input_node in graph.input:
        shape = [int(dim.dim_value) if dim.dim_value > 0 else -1
                 for dim in input_node.type.tensor_type.shape.dim]
        input_info[input_node.name] = shape
    
    # Extract output shapes and names
    output_info = {}
    output_names = []
    for output_node in graph.output:
        shape = [int(dim.dim_value) if dim.dim_value > 0 else -1
                 for dim in output_node.type.tensor_type.shape.dim]
        output_info[output_node.name] = shape
        output_names.append(output_node.name)
    
    print(f"   Input: {input_info}")
    print(f"   Outputs: {output_names}")
    print(f"   Output shapes: {output_info}")
    
    # Save metadata for A3 tensor parsing
    metadata = {
        'input_shapes': input_info,
        'output_shapes': output_info,
        'output_names': output_names,
        'onnx_file': str(onnx_path.name),
        'onnx_size_mb': round(onnx_size_mb, 2),
        'export_params': export_params,
        'imgsz': 320,
    }
    
    metadata_path = Path('onnx_info.txt')
    with open(metadata_path, 'w') as f:
        f.write("=" * 75 + "\n")
        f.write("ONNX MODEL METADATA — For A3 Tensor Parsing\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"File: {onnx_path.name}\n")
        f.write(f"Size: {onnx_size_mb:.2f} MB\n")
        f.write(f"Input Size (imgsz): 320×320×3 (BGR)\n\n")
        f.write("INPUT SHAPES:\n")
        for name, shape in input_info.items():
            f.write(f"  {name}: {shape}\n")
        f.write("\nOUTPUT LAYERS:\n")
        for name in output_names:
            f.write(f"  {name}: {output_info[name]}\n")
        f.write("\nEXPORT PARAMETERS:\n")
        for k, v in export_params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nNOTE: Share this file with team — needed for A3 detection parsing\n")
    
    print(f"   ✓ Metadata saved to: {metadata_path.name}")
    
    print("\n" + "✓ A1 COMPLETE".center(75))
    print(f"  yolo11n.onnx ({onnx_size_mb:.2f} MB) ready for blob conversion")
    
    return str(onnx_path), metadata


def convert_onnx_to_blob(onnx_path):
    """
    A2: Convert ONNX → Blob using blobconverter (or fallback instructions).
    Returns: blob_path (str) if successful, None if manual conversion required
    """
    print_section("A2: CONVERT ONNX → BLOB (MYRIADX)")
    
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    print("\n1. Checking blobconverter availability...")
    try:
        import blobconverter
        print("   ✓ blobconverter found")
    except ImportError:
        print("   ✗ blobconverter not installed")
        print("   Installing: pip install blobconverter...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'blobconverter', '-q'])
            import blobconverter
            print("   ✓ blobconverter installed successfully")
        except Exception as e:
            print(f"   ✗ Auto-install failed: {e}")
            return None
    
    print("\n2. Attempting automated conversion (may take 30-120 seconds)...")
    print("   Uploading ONNX and requesting conversion from OpenVINO backend...")
    
    try:
        blob_path = blobconverter.from_onnx(
            model=onnx_path,
            data_type="FP16",
            shaves=6,               # OAK-D has 16 shaves; 6 is optimal
            use_cache=False,
            version="2022.1",       # OpenVINO 2022.1 for MyriadX
            output_dir=str(models_dir),
        )
        
        blob_file = Path(blob_path)
        
        if not blob_file.exists():
            print(f"   ✗ Conversion returned path but file missing: {blob_path}")
            return None
        
        blob_size_mb = blob_file.stat().st_size / (1024**2)
        
        if blob_size_mb < 2:
            print(f"   ⚠️  WARNING: Blob is very small ({blob_size_mb:.2f} MB)")
            print("      Conversion may have failed. Try manual method.")
            return None
        
        print(f"   ✓ Conversion successful!")
        print(f"   ✓ Blob saved: {blob_file.name} ({blob_size_mb:.2f} MB)")
        
        print("\n" + "✓ A2 COMPLETE".center(75))
        print(f"  yolo11n.blob ({blob_size_mb:.2f} MB) ready for OAK-D")
        
        return str(blob_path)
        
    except Exception as e:
        print(f"   ✗ Conversion failed: {e}")
        return None


def print_manual_conversion_guide(onnx_path):
    """Print manual conversion guide when automated method fails."""
    print("\n" + "!" * 75)
    print("AUTOMATED CONVERSION FAILED — MANUAL FALLBACK".center(75))
    print("!" * 75)
    
    print("\nMANUAL CONVERSION VIA WEB TOOL (takes ~2-5 minutes):\n")
    print("  1. Open browser: http://tools.luxonis.com")
    print(f"  2. Upload file: {Path(onnx_path).name}")
    print("  3. Select conversion options:")
    print("     • Target device: MyriadX")
    print("     • Data type: FP16")
    print("     • Shaves: 6")
    print("     • OpenVINO version: 2022.1")
    print("  4. Click 'Convert' button (wait ~2-5 minutes)")
    print("  5. Download the .blob file")
    print("  6. Save to: ./models/yolo11n.blob")
    print("\nVerify blob size after download:")
    print("  python3 -c \"from pathlib import Path; ")
    print("  size = Path('models/yolo11n.blob').stat().st_size / (1024**2); ")
    print("  print(f'Blob size: {size:.1f} MB')\"")
    print("\n" + "!" * 75 + "\n")


def generate_requirements_file():
    """Generate requirements_export.txt for team reproducibility."""
    print("3. Generating requirements_export.txt...")
    
    requirements = """# YOLO Export to Blob — Requirements
# Install with: pip install -r requirements_export.txt

# Core YOLO framework (auto-downloads yolo11n.pt on first run)
ultralytics>=8.2.0

# ONNX export & validation
onnx>=1.14.0
onnxsim>=0.4.0

# Automated blob conversion
blobconverter>=0.6.7

# Optional: OpenVINO tools (for advanced debugging)
openvino>=2022.3
"""
    
    req_path = Path('requirements_export.txt')
    with open(req_path, 'w') as f:
        f.write(requirements)
    
    print(f"   ✓ Created: {req_path.name}")


def generate_copy_to_pi_script():
    """Generate copy_to_pi.sh for deploying models to Raspberry Pi."""
    print("4. Generating copy_to_pi.sh (deployment to Pi)...")
    
    script = """#!/bin/bash
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
rsync -avz --progress \\
    --include="*.blob" \\
    --include="*.md" \\
    --include="onnx_info.txt" \\
    --exclude="*" \\
    models/ \\
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
"""
    
    script_path = Path("copy_to_pi.sh")
    with open(script_path, 'w') as f:
        f.write(script)
    
    # Make executable on Unix
    if os.name != 'nt':
        os.chmod(script_path, 0o755)
    
    print(f"   ✓ Created: {script_path.name}")
    if os.name != 'nt':
        print(f"      (executable on Linux/macOS)")


def print_final_checklist(onnx_path, blob_path, metadata):
    """Print final completion checklist."""
    print_section("✨ FINAL CHECKLIST — A1 + A2")
    
    onnx_file = Path(onnx_path)
    onnx_size = onnx_file.stat().st_size / (1024**2)
    
    print(f"\nFILES GENERATED:")
    print(f"  [✓] yolo11n.onnx ({onnx_size:.2f} MB)")
    print(f"  [✓] onnx_info.txt")
    print(f"  [✓] requirements_export.txt")
    print(f"  [✓] copy_to_pi.sh")
    
    if blob_path:
        blob_file = Path(blob_path)
        blob_size = blob_file.stat().st_size / (1024**2)
        print(f"  [✓] models/yolo11n.blob ({blob_size:.2f} MB)")
        
        print(f"\nSTATUS: ✓ READY FOR PI")
        print(f"\nNEXT STEPS:")
        print(f"  1. Verify blob exists: ls -lh models/")
        print(f"  2. Share models/ with team (upload to shared storage)")
        print(f"  3. Deploy to Pi: bash copy_to_pi.sh 192.168.1.100")
        print(f"  4. On Pi verify: ls -lh ~/safety_rover/models/")
        print(f"  5. Update oak_pipeline.py with: blob_path='models/yolo11n.blob'")
    else:
        print(f"  [✗] models/yolo11n.blob (MANUAL CONVERSION NEEDED)")
        
        print(f"\nSTATUS: ⚠️  AWAITING MANUAL CONVERSION")
        print(f"\nNEXT STEPS:")
        print(f"  1. See manual conversion guide above (http://tools.luxonis.com)")
        print(f"  2. After downloading .blob, save to: models/yolo11n.blob")
        print(f"  3. Verify: python3 -c \"from pathlib import Path; ")
        print(f"     print(Path('models/yolo11n.blob').stat().st_size / (1024**2))\"")
        print(f"  4. Then deploy: bash copy_to_pi.sh 192.168.1.100")
    
    print(f"\n" + "=" * 75)
    print(f"OUTPUT METADATA (saved in onnx_info.txt):")
    print(f"=" * 75)
    print(f"  Input shape: {metadata['input_shapes']}")
    print(f"  Output names: {metadata['output_names']}")
    print(f"  Output shapes: {metadata['output_shapes']}")
    print(f"\nShare onnx_info.txt with Person C for A3 tensor parsing\n")


def main():
    """Main execution: A1 + A2 complete pipeline."""
    try:
        print("\n" + "🚀 YOLO11n → Blob Conversion Pipeline".center(75))
        print("A1: YOLO → ONNX | A2: ONNX → Blob".center(75))
        
        # Step 0: Check dependencies
        check_and_install_dependencies()
        
        # Step 1: Export YOLO to ONNX (A1)
        onnx_path, metadata = export_yolo_to_onnx()
        
        # Step 2: Convert ONNX to Blob (A2)
        blob_path = convert_onnx_to_blob(onnx_path)
        
        # If automated conversion failed, show manual guide
        if blob_path is None:
            print_manual_conversion_guide(onnx_path)
        
        # Step 3: Generate supporting files
        print_section("STEP 3: GENERATE DEPLOYMENT FILES")
        generate_requirements_file()
        generate_copy_to_pi_script()
        
        # Step 4: Print final summary
        print_final_checklist(onnx_path, blob_path, metadata)
        
        print("\n✨ Done! Ready for next phase.\n")
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
