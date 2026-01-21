# Technical Concerns

**Last Updated**: 2026-01-21
**Codebase**: AI Badminton Coaching System v2.0

---

## Overview

This document identifies technical debt, known issues, fragile areas, and security concerns in the codebase.

---

## Critical Issues

### 1. Protobuf Version Dependency (CRITICAL)

**Location**: System-wide (MediaPipe dependency)
**Severity**: Critical - System breaks without fix

**Problem**:
```
MediaPipe 0.10.9 requires protobuf 3.x
Default pip install often gets protobuf 4.x
Causes mutex blocking error: [mutex.cc : 452] RAW: Lock blocking
```

**Impact**: Complete system failure, cryptic error message

**Workaround**: Manual intervention required
```bash
pip uninstall protobuf -y
pip install protobuf==3.20.3
```

**Root cause**: Version conflict not expressed in `requirements.txt`

**Fix recommendation**:
- Add `protobuf==3.20.3` to `requirements.txt` (pinned version)
- Add validation to `diagnose.py` (already implemented)
- Document prominently in README (already documented)

**Why not fixed**: Requires coordinated update of requirements.txt

---

### 2. Large File in Git History

**Location**: `data/processed/splits/train_data.pkl` (removed, but in history)
**Severity**: High - Blocks GitHub push

**Problem**:
- 105 MB file was committed to git
- Exceeds GitHub's 100 MB file size limit
- Prevents pushing to remote repository

**Status**: **RESOLVED** - File removed from git history using `git filter-branch`

**Actions taken** (2026-01-21):
1. Ran `git filter-branch` to remove `train_data.pkl` from all commits
2. Cleaned up backup refs: `rm -rf .git/refs/original/`
3. Garbage collected: `git gc --prune=now --aggressive`
4. Verified file removed from history

**Remaining concern**: Requires force push to update remote
```bash
git push origin <branch> --force
```

**Prevention**: `.gitignore` already excludes `data/processed/splits/*.pkl`

---

## High Priority Issues

### 3. No Automated Testing

**Location**: Entire codebase
**Severity**: High - No regression detection

**Problem**:
- Zero automated tests (no pytest, unittest)
- Manual testing only (time-consuming, error-prone)
- No CI/CD pipeline
- High risk of regressions during refactoring

**Impact**:
- Difficult to refactor safely
- No confidence in changes
- Hidden bugs in edge cases

**Evidence**: See [`TESTING.md`](.planning/codebase/TESTING.md) for full analysis

**Recommendation**: Start with unit tests for feature engineering (highest ROI)

**Why not addressed**: Time/priority tradeoff during development

---

### 4. Hardcoded Magic Numbers

**Location**: Throughout codebase
**Severity**: Medium - Maintainability issue

**Examples**:
```python
# src/coaching/feedback_generator.py
if distance > 0.02:  # What does 0.02 mean?
    severity = 'critical'

# analyze_video.py
if max_velocity > 90:  # Why 90?
    smash_score += 3
elif max_velocity > 80:
    smash_score += 2

# src/data_processing/extract_poses.py
ARM_ELEVATION_THRESHOLD = 0.3   # Why 0.3?
MIN_VALID_FRAMES_PERCENTAGE = 50  # Why 50%?
```

**Impact**:
- Hard to understand intent
- Difficult to tune parameters
- No single source of truth

**Recommendation**:
- Extract to named constants with comments
- Create `config.py` module for all thresholds
- Document rationale for each value

---

### 5. Root-Level Clutter

**Location**: Project root directory
**Severity**: Medium - Organization issue

**Problem**: 10+ debug/utility scripts at root level
```
analyze_stroke_types.py
analyze_video.py
baseline_model_debug.py
baseline_model_fixed.py
create_forehand_features_pkl.py
debug_splits_colab.py
diagnose.py
filter_backhand_and_regenerate.py
fix_baseline_paths.py
regenerate_features.py
```

**Impact**:
- Hard to distinguish entry points from utilities
- Cluttered repository structure
- Unclear which scripts are production vs debug

**Recommendation**:
- Create `scripts/` directory for utilities
- Keep only `analyze_video.py` and `diagnose.py` at root
- Move others to `scripts/debug/` or `scripts/utils/`

---

## Medium Priority Issues

### 6. Unused ML Infrastructure

**Location**: `src/models/`, `experiments/`, `mlruns/`
**Severity**: Medium - Code clutter, confusing

**Problem**: v2.0 pivoted from ML to benchmark-based analysis
- `src/models/baseline_model.py` - Unused classifier
- `src/models/lstm_model.py` - Unused LSTM model
- `experiments/` directory - Empty
- `mlruns/` directory - MLflow tracking (inactive)
- `data/processed/splits/` - Train/val/test splits (unused)

**Impact**:
- Confuses new contributors
- Dead code in codebase
- Wasted storage (split files)

**Recommendation**:
- Archive to `archive/ml_experiments/`
- Update README to clarify v2.0 is benchmark-based
- Remove unused imports (`mlflow` in requirements)

---

### 7. Duplicate Experiment Directories

**Location**: `data/processed/`
**Severity**: Medium - Storage waste, confusion

**Problem**: Multiple experiment variants with similar structure
```
data/processed/
├── poses/
├── poses_drop_smash/      # Drop vs Smash experiment
├── poses_lift_smash/      # Lift vs Smash experiment
├── poses_multiclass/      # Multiclass experiment
├── features/
├── features_drop_smash/
├── features_lift_smash/
├── features_multiclass/
├── splits/
├── splits_drop_smash/
├── splits_lift_smash/
└── splits_multiclass/
```

**Impact**:
- Storage duplication (~1-2 GB)
- Unclear which is production
- Confusing directory structure

**Recommendation**:
- Archive old experiments to `experiments/archive/`
- Keep only `poses/`, `features/`, `clips/`
- Document experiment history in `docs/experiments.md`

---

### 8. No Input Sanitization

**Location**: Entry points (`analyze_video.py`, `streamlit_app.py`)
**Severity**: Medium - Security concern

**Problem**: No validation or sanitization of user input
```python
def analyze_video(video_path: str, stroke_type: str = None):
    video_file = Path(video_path)  # User-controlled path
    if not video_file.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return
    # Direct video processing, no sanitization
```

**Vulnerabilities**:
- Path traversal: User could pass `../../../etc/passwd` (mitigated by `Path.exists()` check)
- Command injection: Not present (no shell commands with user input)
- File type validation: No check for actual video format (relies on OpenCV)

**Impact**:
- Low risk in CLI (local use)
- **High risk in web deployment** (Streamlit/Gradio exposed to internet)

**Recommendation**:
- Validate file extensions (`.mp4`, `.avi`, `.mov`, `.mkv`)
- Check magic bytes (file signature) to verify actual video
- Restrict paths to specific directories
- Add rate limiting for web deployment
- Never expose to public internet without authentication

---

### 9. Inconsistent Error Handling

**Location**: Throughout codebase
**Severity**: Medium - User experience issue

**Problem**: Mix of error handling strategies
1. Try-catch with informative messages ([`analyze_video.py`](analyze_video.py))
2. Silent failures (return early, no error)
3. Print and continue (no exception raised)
4. Raise exceptions (rare)

**Examples**:
```python
# Pattern 1: Good error handling
try:
    extractor = PoseExtractor()
    pose_data = extractor.extract_from_video(video_path, stroke_type)
except Exception as e:
    print(f"❌ Error during pose extraction: {e}")
    print("\nTroubleshooting:")
    print("1. Ensure player is clearly visible...")
    return

# Pattern 2: Silent failure
def _analyze_arm_extension(self, features, benchmarks):
    if metric not in features or metric not in benchmarks:
        return  # Silently skip, no error message

# Pattern 3: Print and continue
if len(positions) < 2:
    print("Warning: Not enough positions for velocity calculation")
    return np.zeros_like(positions)
```

**Impact**:
- Inconsistent user experience
- Hard to debug issues
- Silent failures hide problems

**Recommendation**:
- Standardize error handling strategy
- Use custom exceptions (`InsufficientDataError`, `InvalidVideoError`)
- Log errors to file for debugging
- Provide actionable error messages

---

### 10. MediaPipe Non-Determinism

**Location**: [`src/data_processing/extract_poses.py`](src/data_processing/extract_poses.py)
**Severity**: Medium - Reproducibility issue

**Problem**: Pose estimation is non-deterministic
- Same video can produce slightly different keypoint coordinates
- Affects downstream features, feedback, scores
- No way to guarantee reproducible results

**Why**: MediaPipe uses GPU acceleration, floating-point math, internal heuristics

**Impact**:
- Unit tests would need tolerance ranges
- Regression testing is harder
- User sees different scores for same video on different runs (rare but possible)

**Mitigation**:
- Document non-determinism in README
- Use relative comparisons (rank, percentile) instead of absolute values
- Test ranges, not exact values

**Not a bug**: Inherent to CV/ML models, acceptable for this use case

---

## Low Priority Issues

### 11. Missing Type Hints (Partial Coverage)

**Location**: Throughout codebase
**Severity**: Low - Code quality issue

**Problem**: Type hints present but inconsistent
- Some functions fully typed
- Others partially typed
- Many untyped

**Example**:
```python
# Fully typed
def analyze_technique(self, features: Dict, stroke_type: str) -> List[FeedbackItem]:
    ...

# Partially typed
def create_radar_chart(features, stroke_type: str):
    ...

# Untyped
def extract_from_video(video_path):
    ...
```

**Impact**:
- Harder to use IDE autocomplete
- Type errors not caught by MyPy
- Less self-documenting code

**Recommendation**: Run `mypy --strict` and fix errors incrementally

---

### 12. Backup Files in Codebase

**Location**: [`src/coaching/technique_benchmarks_backup.py`](src/coaching/technique_benchmarks_backup.py)
**Severity**: Low - Code clutter

**Problem**: Backup files committed to git
- `technique_benchmarks_backup.py` - Old version of benchmarks
- Unclear if still needed
- Should use git history instead

**Recommendation**: Delete and rely on git history

---

### 13. Commented-Out Code

**Location**: [`requirements.txt`](requirements.txt), various Python files
**Severity**: Low - Code clutter

**Problem**: Commented dependencies and code blocks
```txt
# requirements.txt
# Alternative: pytorch>=2.0.0  # Uncomment if using PyTorch instead

# Testing (Optional)
# pytest>=7.4.0
# pytest-cov>=4.1.0

# Code Quality (Optional)
# black>=23.7.0
```

**Impact**: Unclear if code should be deleted or kept

**Recommendation**: Either delete or move to separate `requirements-dev.txt`

---

### 14. No Logging Infrastructure

**Location**: Entire codebase
**Severity**: Low - Observability issue

**Problem**: Uses `print()` statements instead of logging
```python
print("✅ Pose detector initialized")
print(f"❌ Error: Video file not found: {video_path}")
```

**Impact**:
- Can't filter by log level (DEBUG, INFO, ERROR)
- Can't redirect to file
- Hard to disable in production

**Recommendation**: Replace with `logging` module
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Pose detector initialized")
logger.error("Video file not found: %s", video_path)
```

---

### 15. Emojis in Output (Accessibility)

**Location**: Throughout user-facing messages
**Severity**: Low - Accessibility concern

**Problem**: Heavy use of emojis in terminal output
```python
print("✅ Analysis complete")
print("❌ Error")
print("🎯 Drill: ...")
print("🤖 Auto-detecting...")
```

**Impact**:
- May not render correctly on all terminals
- Screen readers may not pronounce correctly
- Visually impaired users affected

**Recommendation**: Make emojis configurable (env var or config file)

---

## Security Concerns

### 16. No Authentication (Web Interfaces)

**Location**: [`src/deployment/streamlit_app.py`](src/deployment/streamlit_app.py), [`src/deployment/coaching_app.py`](src/deployment/coaching_app.py)
**Severity**: High (if deployed publicly)

**Problem**: No user authentication or access control
- Anyone with URL can access
- No rate limiting
- No abuse prevention

**Current status**: Localhost-only deployment (mitigates risk)

**If deploying publicly**: MUST add authentication
- Options: HTTP Basic Auth, OAuth, API keys
- Add rate limiting (max N requests per IP)
- Add HTTPS
- Sanitize all inputs

---

### 17. No HTTPS (Web Interfaces)

**Location**: Web deployments
**Severity**: Medium (if deployed on network)

**Problem**: HTTP only, no encryption
- Streamlit: `http://localhost:8501`
- Gradio: `http://localhost:7860`

**Impact**:
- Videos transmitted in plaintext
- Vulnerable to man-in-the-middle attacks (if on shared network)

**Mitigation**: Deploy behind reverse proxy (nginx) with HTTPS

---

### 18. OpenAI API Key in Environment

**Location**: [`src/coaching/llm_enhancer.py`](src/coaching/llm_enhancer.py)
**Severity**: Low (feature optional)

**Problem**: API key stored in plaintext environment variable
```python
self.api_key = api_key or os.getenv("OPENAI_API_KEY")
```

**Impact**:
- If env vars leaked, API key exposed
- No key rotation mechanism

**Recommendation**:
- Use secrets management (Vault, AWS Secrets Manager)
- Rotate keys regularly
- Monitor API usage for abuse

---

## Performance Concerns

### 19. Synchronous Video Processing

**Location**: [`analyze_video.py`](analyze_video.py), entry points
**Severity**: Medium - Scalability issue

**Problem**: Single-threaded, synchronous processing
- Can only process one video at a time
- Long videos block entire pipeline
- No parallelization

**Impact**:
- Poor user experience for batch processing
- Underutilized CPU (8 cores idle)

**Recommendation**:
- Add batch processing mode (`analyze_videos.py` for multiple files)
- Use `multiprocessing` to parallelize
- Consider async/await for I/O-bound ops

---

### 20. No Caching

**Location**: Entire pipeline
**Severity**: Low - User experience issue

**Problem**: No caching of intermediate results
- Re-running same video repeats all stages
- Pose extraction is expensive (~10s per video)
- Feature engineering is fast but redundant

**Impact**: Wasted computation, slower feedback iteration

**Recommendation**:
- Cache pose data: `outputs/cache/poses/<video_hash>.pkl`
- Cache features: `outputs/cache/features/<video_hash>.pkl`
- Invalidate on version change (hash code version)

---

## Data Quality Concerns

### 21. Forehand-Only Benchmarks

**Location**: [`src/coaching/technique_benchmarks.py`](src/coaching/technique_benchmarks.py)
**Severity**: Medium - Feature limitation

**Problem**: Benchmarks derived from forehand strokes only
```python
"""
Professional Technique Benchmarks - FOREHAND ONLY

Updated: 2026-01-16 21:32:34
Source: ShuttleSet dataset (forehand strokes only, backhand filtered out)
Sample sizes:
  - Clear: 427 features
  - Smash: 427 features
"""
```

**Impact**:
- Backhand strokes analyzed against forehand benchmarks (incorrect)
- No stroke-side detection in pipeline
- Feedback may be misleading for backhand strokes

**Recommendation**:
- Add stroke-side detection (left vs right arm active)
- Derive separate backhand benchmarks
- Warn user if stroke side unclear

---

### 22. Limited Stroke Types

**Location**: System-wide
**Severity**: Low - Feature scope

**Problem**: Only supports Clear and Smash
- No Drop, Drive, Net, Smash, or Lift
- Auto-detection only works for Clear/Smash
- Benchmarks missing for other strokes

**Impact**: Limited usefulness for complete technique analysis

**Status**: Expected limitation for v2.0 scope

**Future work**: Expand to other stroke types

---

## Summary

### Critical (Must Fix)
1. ✓ **Protobuf version dependency** - Documented, workaround available
2. ✓ **Large file in git history** - **RESOLVED** (removed from history)

### High Priority (Should Fix Soon)
3. **No automated testing** - Highest technical debt
4. **Hardcoded magic numbers** - Maintainability issue
5. **Root-level clutter** - Organization issue

### Medium Priority (Address Eventually)
6. Unused ML infrastructure
7. Duplicate experiment directories
8. No input sanitization (critical if deployed publicly)
9. Inconsistent error handling
10. MediaPipe non-determinism (document, not fix)

### Low Priority (Nice to Have)
11-15. Type hints, backup files, commented code, logging, emojis

### Security (Context-Dependent)
16-18. Authentication, HTTPS, API keys (low risk for localhost, high for production)

### Performance (Optimization Opportunities)
19-20. Synchronous processing, no caching

### Data Quality (Known Limitations)
21-22. Forehand-only benchmarks, limited stroke types

---

## Risk Assessment

**Overall risk level**: Medium
- Core functionality works well
- Main risks are around deployment and testing
- No critical security vulnerabilities for local use
- **Must address** before public deployment

**Recommended priority**:
1. Add automated tests (high impact, prevents regressions)
2. Force push to resolve git history issue
3. Add input validation (especially for web deployment)
4. Refactor magic numbers into config
5. Clean up unused code
