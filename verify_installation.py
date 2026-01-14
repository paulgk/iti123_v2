#!/usr/bin/env python3
"""
Verify Installation Script
Check if all required libraries for ITI123 project are properly installed
"""

import sys

def check_import(package_name, import_name=None):
    """Try to import a package and report status"""
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name:20s} - version {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name:20s} - NOT FOUND ({str(e)})")
        return False
    except Exception as e:
        print(f"⚠️  {package_name:20s} - ERROR: {str(e)}")
        return False

print("=" * 60)
print("ITI123 Project - Installation Verification")
print("=" * 60)
print()

print("Python Version:")
print(f"  {sys.version}")
print()

print("Checking Core Libraries:")
print("-" * 60)

required_packages = [
    ("TensorFlow", "tensorflow"),
    ("NumPy", "numpy"),
    ("Pandas", "pandas"),
    ("OpenCV", "cv2"),
    ("MediaPipe", "mediapipe"),
    ("Scikit-learn", "sklearn"),
    ("Matplotlib", "matplotlib"),
    ("Seaborn", "seaborn"),
    ("Gradio", "gradio"),
    ("MLflow", "mlflow"),
    ("Jupyter", "jupyter"),
]

all_ok = True
for display_name, import_name in required_packages:
    result = check_import(display_name, import_name)
    all_ok = all_ok and result

print()
print("=" * 60)

if all_ok:
    print("✅ ALL REQUIRED LIBRARIES INSTALLED SUCCESSFULLY!")
    print()
    print("Next steps:")
    print("1. Place video files in: data/raw_videos/")
    print("2. Download ShuttleSet annotations to: data/annotations/")
    print("3. Start Jupyter: jupyter notebook")
    print("4. Begin with: notebooks/01_data_exploration.ipynb")
else:
    print("⚠️  SOME LIBRARIES MISSING OR HAVE ERRORS")
    print()
    print("Please run: pip install -r requirements.txt")

print("=" * 60)
print()

# Additional checks
print("Testing Key Functionality:")
print("-" * 60)

# Test TensorFlow
try:
    import tensorflow as tf
    print(f"✅ TensorFlow GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print(f"   GPU devices: {[d.name for d in tf.config.list_physical_devices('GPU')]}")
    else:
        print("   Note: Running on CPU (GPU will speed up training)")
except:
    print("⚠️  Could not check TensorFlow GPU")

# Test OpenCV
try:
    import cv2
    print(f"✅ OpenCV can read videos: {hasattr(cv2, 'VideoCapture')}")
except:
    print("⚠️  Could not verify OpenCV video capability")

# Test MediaPipe
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    print(f"✅ MediaPipe Pose available: {mp_pose is not None}")
except:
    print("⚠️  Could not verify MediaPipe Pose")

print("=" * 60)
