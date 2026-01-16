#!/usr/bin/env python3
"""
Fix baseline_model.py paths for Colab

This script patches baseline_model.py to use correct paths in Colab.
"""

import sys
from pathlib import Path

# Read the original file
baseline_file = Path("src/models/baseline_model.py")

with open(baseline_file, 'r') as f:
    content = f.read()

# Find and replace the BASE_DIR line
old_line = "BASE_DIR = Path(__file__).resolve().parents[2]"
new_line = """# Auto-detect if running in Colab
import os
if 'COLAB_GPU' in os.environ or '/content/' in os.getcwd():
    # Running in Colab
    BASE_DIR = Path("/content/drive/MyDrive/ITI123")
else:
    # Running locally
    BASE_DIR = Path(__file__).resolve().parents[2]"""

if old_line in content:
    content = content.replace(old_line, new_line)
    print("✓ Patched BASE_DIR to auto-detect Colab")
else:
    print("⚠️  Could not find BASE_DIR line to patch")
    sys.exit(1)

# Write back
with open(baseline_file, 'w') as f:
    f.write(content)

print(f"✓ Updated {baseline_file}")
print()
print("Now run in Colab:")
print("  !python src/models/baseline_model.py")
