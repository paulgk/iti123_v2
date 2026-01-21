# Code Conventions

**Last Updated**: 2026-01-21
**Codebase**: AI Badminton Coaching System v2.0

---

## Code Style Overview

**General approach**: Mostly follows PEP 8 conventions with some inconsistencies. No automated formatters (Black, autopep8) or linters (Flake8) in active use.

---

## Naming Conventions

### Files & Modules
- **Snake_case**: `extract_poses.py`, `feature_engineering_v2.py`, `feedback_generator.py`
- **Version suffixes**: `_v2` for second iterations (`feature_engineering_v2.py`)
- **Backup suffix**: `_backup` for old versions (`technique_benchmarks_backup.py`)
- **Descriptive names**: Clear purpose from filename alone

**Examples**:
```
src/data_processing/extract_poses.py          # Good: verb + noun
src/coaching/feedback_generator.py             # Good: noun + generator pattern
src/data_processing/feature_engineering_v2.py  # Version suffix for iteration
```

### Classes
- **PascalCase**: `PoseExtractor`, `CoachingFeedback`, `TechniqueVisualizer`, `FeedbackItem`
- **Descriptive suffixes**: `Generator`, `Visualizer`, `Extractor`, `Item`
- **No prefixes**: Avoid Hungarian notation (no `CMyClass`, `IInterface`)

**Examples**:
```python
class PoseExtractor:           # Good: noun + action suffix
class CoachingFeedback:        # Good: domain noun
class FeedbackItem:            # Good: item suffix for data containers
class LLMCoachingEnhancer:     # Good: descriptive, clear purpose
```

### Functions & Methods
- **Snake_case**: `extract_from_video()`, `analyze_technique()`, `get_benchmarks()`
- **Verb-first for actions**: `extract_`, `calculate_`, `analyze_`, `create_`
- **Private methods**: Single underscore prefix `_analyze_velocity()`
- **Boolean predicates**: No consistent `is_` or `has_` prefix (inconsistent)

**Examples**:
```python
def extract_from_video(video_path):        # Good: verb + from + noun
def analyze_technique(features, stroke):   # Good: verb + noun
def _analyze_velocity(features, bench):    # Good: private with underscore
def calculate_velocity(positions):         # Good: verb + noun
def get_benchmarks(stroke_type):           # Good: get + noun
```

**Inconsistencies**:
```python
# Missing boolean prefixes
def detect_stroke_type(features):   # Could be: is_smash() or returns type
```

### Variables
- **Snake_case**: `pose_data`, `stat_features`, `overall_score`, `stroke_type`
- **Descriptive over concise**: Favor clarity (`max_velocity` vs `max_v`)
- **No single-letter except loops**: Avoid `x`, `y` except in math contexts

**Examples**:
```python
pose_data = {...}                    # Good: noun + noun
stat_features = {...}                # Good: adjective + noun
overall_score = 85                   # Good: descriptive
stroke_type = 'Clear'                # Good: clear purpose

# Loop variables (acceptable)
for i, item in enumerate(items):    # Standard enumeration
for frame in frames:                 # Descriptive loop variable
```

### Constants
- **UPPER_SNAKE_CASE**: `MODEL_COMPLEXITY`, `MIN_DETECTION_CONFIDENCE`, `LANDMARK_RIGHT_WRIST`
- **Descriptive prefixes**: `MIN_`, `MAX_`, `LANDMARK_`, `CLEAR_`, `SMASH_`
- **Module-level placement**: Defined at top of file after imports

**Examples**:
```python
# Configuration constants
MODEL_COMPLEXITY = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_KEYPOINT_CONFIDENCE = 0.3

# Landmark indices
LANDMARK_NOSE = 0
LANDMARK_RIGHT_WRIST = 16

# Benchmarks
CLEAR_BENCHMARKS = {...}
SMASH_BENCHMARKS = {...}
```

### Dictionary Keys
- **Snake_case strings**: `'max_velocity'`, `'elbow_angle_mean'`, `'r_arm_extension'`
- **Prefix conventions**:
  - `r_` for right side: `'r_wrist_x'`, `'r_elbow_angle'`
  - `l_` for left side: `'l_wrist_x'`, `'l_elbow_angle'`
  - `_mean`, `_std`, `_min`, `_max` suffixes for statistics
  - `_target`, `_lower`, `_upper` for benchmark ranges

**Examples**:
```python
features = {
    'max_velocity': 78.5,              # Metric name
    'elbow_angle_mean': 125.3,         # Metric + statistic
    'r_wrist_height_from_head': 0.03,  # Side + metric + reference
    'velocity_p25': 65.2,              # Metric + percentile
}

benchmarks = {
    'max_velocity': 48.15,             # Lower bound
    'max_velocity_target': 75.78,      # Target (median)
    'max_velocity_upper': 92.33,       # Upper bound
}
```

---

## Type Hints

**Usage**: Partial adoption
- **Function signatures**: Many functions have type hints, but not all
- **Return types**: Often specified (`: List[FeedbackItem]`, `: Dict`)
- **Variable annotations**: Rare, mostly untyped

**Examples**:
```python
# Good: Full type annotations
def analyze_technique(self, features: Dict, stroke_type: str) -> List[FeedbackItem]:
    ...

def format_value(value: float, metric: str) -> str:
    ...

# Common pattern: Typing imports
from typing import Dict, List, Tuple, Optional
```

**Inconsistencies**:
```python
# Missing type hints
def extract_from_video(video_path):  # No annotations
    ...

# Partial hints
def create_radar_chart(features, stroke_type: str):  # Only second param
    ...
```

**Recommendation**: Run MyPy for full type coverage (currently commented in requirements)

---

## Docstrings

**Format**: Google-style docstrings (mostly)
- **Module docstrings**: Present, descriptive, often multi-line with metadata
- **Class docstrings**: Present, concise
- **Function docstrings**: Present for public functions, often missing for private methods
- **Sections**: `Args`, `Returns`, `Example`, `References`

**Examples**:

**Module docstring** (comprehensive):
```python
"""
Feature Engineering V2 - Improved Features for Clear vs Smash Classification

Dataset:
    ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton
    Tactical Analysis (Wang et al., 2023)
    https://arxiv.org/abs/2306.04948

Key improvements based on diagnostic analysis:
1. Focus on Z-coordinate (depth) - shows medium effect size
2. Arm extension patterns - shows medium effect size
...

References:
    Wang, W.-Y., Huang, Y.-C., Ik, T.-U., & Peng, W.-C. (2023).
    ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for
    Badminton Tactical Analysis. CoRR, abs/2306.04948.
"""
```

**Function docstring** (Google-style):
```python
def analyze_technique(self, features: Dict, stroke_type: str) -> List[FeedbackItem]:
    """
    Analyze technique and generate coaching feedback

    Args:
        features: Dictionary of biomechanical features (from feature_engineering_v2.py)
        stroke_type: 'Clear', 'Smash', or equivalent

    Returns:
        List of FeedbackItem objects, ordered by severity

    Example:
        >>> feedback_gen = CoachingFeedback()
        >>> items = feedback_gen.analyze_technique(features, 'Clear')
        >>> for item in items[:3]:  # Show top 3 issues
        ...     print(item.format())
    """
```

**Class docstring**:
```python
class FeedbackItem:
    """Single piece of coaching feedback"""
```

**Missing docstrings**:
```python
def _analyze_velocity(self, features, benchmarks):
    # Private method, no docstring (common pattern)
```

---

## Comments

**Style**: Inline comments using `#`, often on same line or above code
**Usage**: Frequent, explanatory, sometimes redundant

**Section headers** (ASCII art):
```python
# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parents[2]
CLIPS_DIR = BASE_DIR / "data" / "processed" / "clips"

# =============================================================================
# POSE DETECTION CONFIGURATION
# =============================================================================
MODEL_COMPLEXITY = 2
```

**Inline comments** (common):
```python
severity_order = {'critical': 0, 'major': 1, 'minor': 2, 'good': 3}  # Sort by severity
self.feedback_items.sort(key=lambda x: severity_order.get(x.severity, 4))

# Get benchmarks for this stroke type
benchmarks = TechniqueBenchmarks.get_benchmarks(stroke_type)
```

**TODOs**: Rare, no consistent format
```python
# TODO: Implement multi-person detection fully
# IMPORTANT: The focused stroke is ALWAYS the FIRST shot in the clip
```

---

## Code Organization

### Module Structure
**Standard order**:
1. Module docstring
2. Imports (stdlib → third-party → local)
3. Constants
4. Functions/Classes
5. Main block (`if __name__ == "__main__"`)

**Example**:
```python
"""Module docstring"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

from src.coaching import TechniqueBenchmarks

# Constants
BASE_DIR = Path(__file__).resolve().parent
MODEL_COMPLEXITY = 2

# Classes/Functions
class PoseExtractor:
    ...

# Main
if __name__ == "__main__":
    main()
```

### Import Organization
**Grouping**: Generally follows PEP 8
```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp

# Local/relative
from src.coaching import TechniqueBenchmarks, CoachingFeedback
from .technique_benchmarks import TechniqueBenchmarks  # Relative import
```

**Inconsistencies**: Some files mix import styles

### Class Organization
**Standard order** (mostly followed):
1. `__init__` method
2. Public methods
3. Private methods (underscore prefix)
4. Special methods (`__repr__`, `__str__`)

**Example**:
```python
class FeedbackItem:
    def __init__(self, metric, severity, ...):
        ...

    def format(self) -> str:  # Public method
        ...

    def __repr__(self):       # Special method
        ...
```

---

## Error Handling

**Pattern**: Try-catch at high level (entry points), minimal elsewhere
**Philosophy**: Fail loudly, print errors, return early

**Common pattern**:
```python
try:
    extractor = PoseExtractor()
    pose_data = extractor.extract_from_video(video_path, stroke_type)
    extractor.close()
except Exception as e:
    print(f"❌ Error during pose extraction: {e}")
    print("\nTroubleshooting:")
    print("1. Ensure player is clearly visible...")
    return  # Early exit
```

**Defensive checks**:
```python
if metric not in features or metric not in benchmarks:
    return  # Silently skip

if len(positions) < 2:
    return np.zeros_like(positions)  # Return safe default

if not video_file.exists():
    print(f"❌ Error: Video file not found: {video_path}")
    return
```

**No custom exceptions**: All exceptions are built-in (`Exception`, `ValueError`, `ImportError`)

---

## Function Design

### Function Length
- **Short functions**: Preferred (10-50 lines)
- **Long functions**: Some analysis functions reach 100+ lines
- **No strict limit**: Pragmatic approach

### Parameters
- **Positional**: Common for 1-3 params
- **Keyword**: Used for optional params
- **Defaults**: Often used (`stroke_type: str = None`, `smooth=True`)

**Example**:
```python
def extract_temporal_features(pose_sequence, smooth=True):
    ...

def analyze_technique(self, features: Dict, stroke_type: str) -> List[FeedbackItem]:
    ...
```

### Return Values
- **Single return**: Most functions return one value
- **Tuples**: Used for multiple returns (`detected_stroke, confidence`)
- **None returns**: Common for side-effect functions (visualization, file I/O)
- **Early returns**: Frequent for error cases and edge conditions

**Examples**:
```python
def detect_stroke_type(features):
    # ... logic ...
    return detected_stroke, confidence  # Tuple

def create_radar_chart(features, stroke_type):
    # ... create chart ...
    plt.savefig(output_path)
    # No return (side effect)

def get_benchmarks(stroke_type):
    if stroke_type not in ['Clear', 'Smash']:
        return None  # Early return for invalid input
    return CLEAR_BENCHMARKS if stroke_type == 'Clear' else SMASH_BENCHMARKS
```

---

## String Formatting

**Multiple styles** (inconsistent):
```python
# f-strings (modern, preferred)
print(f"✓ Extracted {num_frames} frames")
output = f"{icon} {self.message}\n"

# Format method
print("Video: {}".format(video_path))

# Percent formatting (rare)
print("Quality: %.1f%%" % quality)

# String concatenation (rare)
output += "   → " + self.impact + "\n"
```

**Recommendation**: Standardize on f-strings (already most common)

---

## Path Handling

**Consistent use of `pathlib.Path`**:
```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # Project root
CLIPS_DIR = BASE_DIR / "data" / "processed" / "clips"

video_file = Path(video_path)
if not video_file.exists():
    ...

output_dir = Path("outputs") / "video_analysis" / video_file.stem
output_dir.mkdir(parents=True, exist_ok=True)
```

**Strength**: Modern, cross-platform approach throughout codebase

---

## Emojis in Output

**Heavy use** for user-facing messages:
```python
print("✅ Pose detector initialized")
print("❌ Error: Video file not found")
print("🤖 Auto-detecting stroke type...")
print("🎯 Drill: Resistance band training")

# Severity icons
severity_icons = {
    'critical': '🔴',
    'major': '⚠️',
    'minor': '💡',
    'good': '✅'
}
```

**Opinion**: Makes CLI friendly, but could be a config option for accessibility

---

## Code Smells & Anti-Patterns

### Present in Codebase

1. **Magic numbers**: Hardcoded thresholds scattered throughout
   ```python
   if distance > 0.02:  # What does 0.02 mean?
   if score >= 85:       # Why 85?
   ```

2. **Long parameter lists**: Some functions have 7+ parameters
   ```python
   def __init__(self, metric, severity, message, current_value, target_range, drill, impact):
   ```

3. **Commented code**: Some dead code left in comments
   ```python
   # Alternative: pytorch>=2.0.0  # Uncomment if using PyTorch instead
   ```

4. **Inconsistent error handling**: Mix of try-catch, early returns, silent failures

5. **Duplicate logic**: Similar analysis patterns repeated across functions

### Not Present (Good!)

1. **No global mutable state**: All state in function params or class instances
2. **No deeply nested conditionals**: Mostly flat structure with early returns
3. **No god objects**: Classes have clear, focused responsibilities
4. **No circular imports**: Clean dependency structure

---

## Best Practices Followed

1. **`if __name__ == "__main__"` guards**: Present in all scripts
2. **Context managers**: Used for file operations, MediaPipe resources
3. **List comprehensions**: Preferred over `map`/`filter`
4. **f-strings**: Modern string formatting (mostly)
5. **Type hints**: Partial but growing adoption
6. **Docstrings**: Present for public APIs
7. **pathlib.Path**: Consistent modern path handling
8. **Early returns**: Avoid deep nesting

---

## Areas for Improvement

1. **Add code formatter**: Black or autopep8 for consistency
2. **Add linter**: Flake8 or Pylint for style enforcement
3. **Add type checker**: MyPy for full type coverage
4. **Reduce magic numbers**: Extract to named constants
5. **Standardize string formatting**: f-strings everywhere
6. **Add docstrings to private methods**: Improve maintainability
7. **Custom exceptions**: Create domain-specific exceptions
8. **Configuration file**: Centralize hardcoded thresholds

---

## Summary

**Overall code quality**: Good
- Readable, well-structured, mostly follows PEP 8
- Clear naming, good use of modern Python features
- Needs automated tooling for consistency
- Some technical debt in magic numbers and error handling

**Strengths**:
- Clean architecture, clear separation of concerns
- Good documentation (docstrings, comments, README)
- Modern Python idioms (pathlib, f-strings, type hints)

**Weaknesses**:
- No automated formatting/linting
- Inconsistent error handling
- Magic numbers scattered throughout
- Partial type hint coverage
