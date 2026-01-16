#!/usr/bin/env python3
"""
Diagnostic Tool - Check MediaPipe Environment

Checks all dependencies and identifies issues.
"""

import sys


def check_python():
    """Check Python version"""
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and 8 <= version.minor <= 10:
        print("  ✓ Compatible version (3.8-3.10 recommended)")
        return True
    elif version.major == 3 and version.minor >= 11:
        print("  ⚠️  Python 3.11+ may have issues with MediaPipe")
        print("     Consider using Python 3.10")
        return True
    else:
        print("  ✗ Incompatible version")
        return False


def check_mediapipe():
    """Check MediaPipe installation"""
    try:
        import mediapipe as mp
        version = mp.__version__
        print(f"MediaPipe: {version}")
        print("  ✓ Installed")
        return True
    except ImportError as e:
        print("MediaPipe: NOT INSTALLED")
        print(f"  ✗ Error: {e}")
        print("  Fix: pip install mediapipe==0.10.9")
        return False
    except Exception as e:
        print(f"MediaPipe: ERROR")
        print(f"  ✗ {e}")
        return False


def check_protobuf():
    """Check protobuf version (critical!)"""
    try:
        import google.protobuf
        version = google.protobuf.__version__
        print(f"Protobuf: {version}")

        if version.startswith('3.'):
            print("  ✓ Compatible version (3.x)")
            return True
        elif version.startswith('4.'):
            print("  ✗ INCOMPATIBLE: MediaPipe requires protobuf 3.x")
            print("  Fix: pip install protobuf==3.20.3")
            return False
        else:
            print(f"  ⚠️  Unknown version: {version}")
            return False

    except ImportError:
        print("Protobuf: NOT INSTALLED")
        print("  ✗ Fix: pip install protobuf==3.20.3")
        return False
    except Exception as e:
        print(f"Protobuf: ERROR - {e}")
        return False


def check_opencv():
    """Check OpenCV installation"""
    try:
        import cv2
        version = cv2.__version__
        print(f"OpenCV: {version}")
        print("  ✓ Installed")
        return True
    except ImportError:
        print("OpenCV: NOT INSTALLED")
        print("  ✗ Fix: pip install opencv-python")
        return False
    except Exception as e:
        print(f"OpenCV: ERROR - {e}")
        return False


def check_numpy():
    """Check NumPy"""
    try:
        import numpy as np
        version = np.__version__
        print(f"NumPy: {version}")
        print("  ✓ Installed")
        return True
    except ImportError:
        print("NumPy: NOT INSTALLED")
        print("  ✗ Fix: pip install numpy")
        return False


def check_pose_extractor():
    """Check if PoseExtractor can be imported"""
    try:
        from src.data_processing.extract_poses import PoseExtractor
        print("PoseExtractor: OK")
        print("  ✓ Can import successfully")
        return True
    except ImportError as e:
        print("PoseExtractor: IMPORT ERROR")
        print(f"  ✗ {e}")
        print("  Make sure you're in the project root directory")
        return False
    except Exception as e:
        print("PoseExtractor: ERROR")
        print(f"  ✗ {e}")
        return False


def check_coaching_modules():
    """Check coaching modules"""
    try:
        from src.coaching import TechniqueBenchmarks, CoachingFeedback, TechniqueVisualizer
        print("Coaching Modules: OK")
        print("  ✓ All modules import successfully")
        return True
    except Exception as e:
        print("Coaching Modules: ERROR")
        print(f"  ✗ {e}")
        return False


def main():
    print("="*70)
    print("AI BADMINTON COACH - DIAGNOSTIC TOOL")
    print("="*70)
    print("\nChecking environment...\n")

    results = {}

    # Check each component
    print("1. Python Version")
    results['python'] = check_python()
    print()

    print("2. MediaPipe")
    results['mediapipe'] = check_mediapipe()
    print()

    print("3. Protobuf (CRITICAL)")
    results['protobuf'] = check_protobuf()
    print()

    print("4. OpenCV")
    results['opencv'] = check_opencv()
    print()

    print("5. NumPy")
    results['numpy'] = check_numpy()
    print()

    print("6. PoseExtractor Module")
    results['pose_extractor'] = check_pose_extractor()
    print()

    print("7. Coaching Modules")
    results['coaching'] = check_coaching_modules()
    print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")
    print()

    if all(results.values()):
        print("✅ ALL CHECKS PASSED")
        print("\nYou should be able to process videos:")
        print("  python extract_only.py video.mp4")
        print("  python analyze_any_video.py video.mp4")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nFailed components:")
        for name, status in results.items():
            if not status:
                print(f"  ✗ {name}")

        print("\nMost Common Fix:")
        if not results.get('protobuf', True):
            print("  pip install protobuf==3.20.3")
        print("\nSee FIX_MEDIAPIPE_MUTEX.md for detailed solutions")

    print()

    # Specific recommendations
    if not results.get('protobuf', True):
        print("="*70)
        print("CRITICAL: PROTOBUF VERSION ISSUE")
        print("="*70)
        print("\nThe mutex error is caused by wrong protobuf version.")
        print("\nFix now:")
        print("  pip uninstall protobuf -y")
        print("  pip install protobuf==3.20.3")
        print()


if __name__ == "__main__":
    main()
