#!/usr/bin/env python3
"""
Verification Script for v1.0 Milestone

Verifies all core scripts are importable and properly structured.
"""

import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

print("="*70)
print("MILESTONE v1.0 VERIFICATION")
print("="*70)
print()

# Test imports
tests = []

print("Testing Core Imports...")
print("-"*70)

# 1. Data Processing
try:
    from src.data_processing import extract_poses
    print("✅ extract_poses.py import successful")
    tests.append(("extract_poses", True))
except Exception as e:
    print(f"❌ extract_poses.py import failed: {e}")
    tests.append(("extract_poses", False))

try:
    from src.data_processing import feature_engineering_v2
    print("✅ feature_engineering_v2.py import successful")
    tests.append(("feature_engineering_v2", True))
except Exception as e:
    print(f"❌ feature_engineering_v2.py import failed: {e}")
    tests.append(("feature_engineering_v2", False))

try:
    from src.data_processing import data_split
    print("✅ data_split.py import successful")
    tests.append(("data_split", True))
except Exception as e:
    print(f"❌ data_split.py import failed: {e}")
    tests.append(("data_split", False))

# 2. Models
try:
    from src.models import baseline_model
    print("✅ baseline_model.py import successful")
    tests.append(("baseline_model", True))
except Exception as e:
    print(f"❌ baseline_model.py import failed: {e}")
    tests.append(("baseline_model", False))

try:
    from src.models import lstm_model
    print("✅ lstm_model.py import successful")
    tests.append(("lstm_model", True))
except Exception as e:
    print(f"❌ lstm_model.py import failed: {e}")
    tests.append(("lstm_model", False))

# 3. Analysis
try:
    from src.analysis import analyze_wrist_features
    print("✅ analyze_wrist_features.py import successful")
    tests.append(("analyze_wrist_features", True))
except Exception as e:
    print(f"❌ analyze_wrist_features.py import failed: {e}")
    tests.append(("analyze_wrist_features", False))

print()
print("="*70)
print("File Structure Verification")
print("="*70)
print()

# Check key files exist
required_files = [
    "src/data_processing/extract_poses.py",
    "src/data_processing/feature_engineering_v2.py",
    "src/data_processing/data_split.py",
    "src/models/baseline_model.py",
    "src/models/lstm_model.py",
    "src/analysis/analyze_wrist_features.py",
    "outputs/reports/ITI123_Milestone_Report.pdf",
    "outputs/reports/ITI123_Milestone_Report.tex",
    "outputs/reports/FINAL_PROJECT_REPORT.md",
    "outputs/reports/wrist_features_cohens_d.csv",
    "VERSION.md",
    "README.md",
    "CLEANUP_SUMMARY.md"
]

file_checks = []
for filepath in required_files:
    full_path = BASE_DIR / filepath
    exists = full_path.exists()
    status = "✅" if exists else "❌"
    print(f"{status} {filepath}")
    file_checks.append((filepath, exists))

print()
print("="*70)
print("SUMMARY")
print("="*70)
print()

passed_imports = sum(1 for _, success in tests if success)
total_imports = len(tests)
print(f"Import Tests: {passed_imports}/{total_imports} passed")

passed_files = sum(1 for _, exists in file_checks if exists)
total_files = len(file_checks)
print(f"File Checks: {passed_files}/{total_files} passed")

print()
if passed_imports == total_imports and passed_files == total_files:
    print("🎉 ALL VERIFICATIONS PASSED - MILESTONE v1.0 READY")
    sys.exit(0)
else:
    print("⚠️  SOME VERIFICATIONS FAILED - SEE ABOVE")
    sys.exit(1)
