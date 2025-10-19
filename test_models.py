#!/usr/bin/env python3
"""
Test script to load both fire and violence models
and diagnose any loading issues.
"""
import os
import sys

print("=" * 60)
print("MODEL LOADING DIAGNOSTIC")
print("=" * 60)

# Test 1: Check model files exist
print("\n[1] Checking model files...")
fire_path = "models/best.pt"
violence_path = "models/violence_model.keras"

if os.path.isfile(fire_path):
    size_mb = os.path.getsize(fire_path) / (1024 * 1024)
    print(f"✓ Fire model found: {fire_path} ({size_mb:.1f} MB)")
else:
    print(f"✗ Fire model NOT found: {fire_path}")

if os.path.isfile(violence_path):
    size_mb = os.path.getsize(violence_path) / (1024 * 1024)
    print(f"✓ Violence model found: {violence_path} ({size_mb:.1f} MB)")
else:
    print(f"✗ Violence model NOT found: {violence_path}")

# Test 2: Check PyTorch availability for fire model
print("\n[2] Checking PyTorch for fire detection...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} installed")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    # Try loading fire model
    print(f"\n[3] Loading fire model (YOLOv5)...")
    try:
        model = torch.hub.load("ultralytics/yolov5", "custom", path=fire_path, trust_repo=True)
        print(f"✓ Fire model loaded successfully!")
        print(f"  Model classes: {model.names}")
    except Exception as e:
        print(f"✗ Fire model loading FAILED:")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError:
    print("✗ PyTorch NOT installed")
    print("  Install with: pip install torch torchvision")

# Test 3: Check TensorFlow availability for violence model
print("\n[4] Checking TensorFlow for violence detection...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} installed")
    
    # Try loading violence model
    print(f"\n[5] Loading violence model (Keras)...")
    try:
        model = tf.keras.models.load_model(violence_path)
        print(f"✓ Violence model loaded successfully!")
        print(f"  Model input shape: {model.input_shape}")
        print(f"  Model output shape: {model.output_shape}")
        model.summary()
    except Exception as e:
        print(f"✗ Violence model loading FAILED:")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError:
    print("✗ TensorFlow NOT installed")
    print("  Install with: pip install tensorflow==2.16.1")
    print("  Note: Requires Python 3.10 or 3.11 (not 3.13)")

# Test 4: Try loading via backend classes
print("\n[6] Testing backend model classes...")
try:
    from backend.app.models.fire import FireModel
    print("✓ FireModel class imported")
    try:
        fire_model = FireModel(weights_path=fire_path, threshold=0.4)
        if fire_model.model is not None:
            print("✓ FireModel initialized successfully")
        else:
            print("⚠ FireModel initialized but model is None (PyTorch missing?)")
    except Exception as e:
        print(f"✗ FireModel initialization failed: {e}")
except Exception as e:
    print(f"✗ Could not import FireModel: {e}")

try:
    from backend.app.models.violence import ViolenceModel
    print("✓ ViolenceModel class imported")
    try:
        violence_model = ViolenceModel(model_path=violence_path, sequence_length=16, threshold=0.6)
        if violence_model.model is not None:
            print("✓ ViolenceModel initialized successfully")
        else:
            print("⚠ ViolenceModel initialized but model is None (TensorFlow missing?)")
    except Exception as e:
        print(f"✗ ViolenceModel initialization failed: {e}")
except Exception as e:
    print(f"✗ Could not import ViolenceModel: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
