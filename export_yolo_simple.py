#!/usr/bin/env python3
"""
Simple YOLO11n export pipeline (Windows-compatible, no Unicode)
A1: YOLO PT -> ONNX
A2: ONNX -> Blob (manual via web tool)
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if ultralytics is installed."""
    try:
        import ultralytics
        print("[OK] ultralytics installed")
        return True
    except ImportError:
        print("[ERROR] ultralytics not found")
        print("Install: pip install ultralytics")
        return False

def export_yolo_to_onnx():
    """Step A1: Export YOLO11n to ONNX."""
    print("\n" + "="*70)
    print("STEP A1: YOLO11n -> ONNX (320x320, opset=12)")
    print("="*70)
    
    try:
        from ultralytics import YOLO
        
        print("\n[INFO] Loading YOLO11n (auto-downloads ~6MB)...")
        model = YOLO('yolo11n.pt')
        
        print("[INFO] Exporting to ONNX...")
        results = model.export(format='onnx', imgsz=320, opset=12)
        
        onnx_path = Path(results)
        if onnx_path.exists():
            size_mb = onnx_path.stat().st_size / (1024**2)
            print(f"[OK] ONNX exported: {onnx_path.name} ({size_mb:.2f} MB)")
            return str(onnx_path)
        else:
            print(f"[ERROR] ONNX file not found: {onnx_path}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def manual_blob_conversion():
    """Step A2: Print manual conversion instructions."""
    print("\n" + "="*70)
    print("STEP A2: ONNX -> Blob (Manual Conversion)")
    print("="*70)
    
    print("\nThe automated blob conversion via blobconverter may fail on Windows.")
    print("Use the web tool instead (2-5 minutes):")
    print("\n1. Open browser: http://tools.luxonis.com")
    print("2. Upload file: yolo11n.onnx")
    print("3. Select conversion options:")
    print("   - Target device: MyriadX")
    print("   - Data type: FP16")
    print("   - Shaves: 6")
    print("   - OpenVINO version: 2022.1")
    print("4. Click 'Convert' and wait")
    print("5. Download: yolo11n.blob")
    print("6. Save to: safety_rover/models/yolo26n.blob")

def verify_onnx():
    """Verify ONNX model structure."""
    print("\n" + "="*70)
    print("VERIFICATION: ONNX Model Structure")
    print("="*70)
    
    try:
        import onnx
        
        onnx_path = Path('yolo11n.onnx')
        if not onnx_path.exists():
            print(f"[ERROR] ONNX file not found: {onnx_path}")
            return
        
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        
        print(f"[OK] ONNX model is valid")
        
        # Extract input/output info
        graph = model.graph
        print(f"\nInputs:")
        for inp in graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  {inp.name}: {shape}")
        
        print(f"\nOutputs:")
        for out in graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  {out.name}: {shape}")
        
        # Save metadata
        with open('onnx_info.txt', 'w') as f:
            f.write("YOLO11n ONNX Model Info\n")
            f.write("="*50 + "\n\n")
            f.write("Inputs:\n")
            for inp in graph.input:
                shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
                f.write(f"  {inp.name}: {shape}\n")
            f.write("\nOutputs:\n")
            for out in graph.output:
                shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
                f.write(f"  {out.name}: {shape}\n")
        
        print("\n[OK] Metadata saved to: onnx_info.txt")
        
    except ImportError:
        print("[WARNING] onnx not installed, skipping validation")
        print("Install: pip install onnx")
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")

def main():
    print("\n" + "="*70)
    print("YOLO11n Export Pipeline - Windows Compatible")
    print("="*70)
    
    # Check dependencies
    if not check_dependencies():
        print("\n[FATAL] Missing dependencies")
        sys.exit(1)
    
    # Step A1: Export to ONNX
    onnx_path = export_yolo_to_onnx()
    if not onnx_path:
        print("\n[FATAL] A1 export failed")
        sys.exit(1)
    
    # Verify ONNX
    try:
        import onnx
        verify_onnx()
    except ImportError:
        print("[INFO] Skipping ONNX verification (onnx not installed)")
    
    # Step A2: Manual blob conversion
    manual_blob_conversion()
    
    # Summary
    print("\n" + "="*70)
    print("EXPORT PIPELINE COMPLETE")
    print("="*70)
    print("\nNext Steps:")
    print("1. Convert ONNX to blob using web tool (see above)")
    print("2. Place blob at: safety_rover/models/yolo26n.blob")
    print("3. Run: python oak_pipeline.py (on Pi)")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
