# Technology Stack

**Last Updated**: 2026-01-21
**Codebase**: AI Badminton Coaching System v2.0

---

## Runtime & Language

**Python 3.8-3.10** (3.10 recommended)
- Primary language for all components
- Supports TensorFlow 2.15.x compatibility requirements
- Used across data processing, ML models, coaching logic, and deployment

---

## Core Deep Learning

**TensorFlow 2.15.x**
- `tensorflow>=2.15.0,<2.16.0` (pinned for MediaPipe compatibility)
- Primary ML framework
- Used for: Model training, inference (though current system is benchmark-based)
- Critical version constraint: Must stay < 2.16 for MediaPipe 0.10.9

**Alternative considered**: PyTorch 2.0+ (commented in requirements, not active)

---

## Computer Vision & Pose Estimation

**MediaPipe 0.10.9** (pinned version - CRITICAL)
- `mediapipe==0.10.9`
- Core technology for pose estimation
- Extracts 33 keypoints from video frames
- Model complexity: 2 (Heavy - most accurate)
- Location: [`src/data_processing/extract_poses.py`](src/data_processing/extract_poses.py)
- **Known issue**: Requires `protobuf==3.20.3` to avoid mutex blocking errors

**OpenCV 4.8+**
- `opencv-python>=4.8.0`
- Video I/O and frame processing
- Supports: .mp4, .avi, .mov, .mkv formats
- Used in: Pose extraction, visualization overlays

---

## Data Processing & ML

**NumPy 1.24+ (< 2.0)**
- `numpy>=1.24.0,<2.0.0`
- Core numerical operations
- Pose keypoint arrays, feature vectors, statistical computations

**Pandas 2.0+**
- `pandas>=2.0.0`
- Metadata management ([`clips_metadata.csv`](data/processed/clips/clips_metadata.csv))
- Feature dataframes, benchmark tables

**scikit-learn 1.3+**
- `scikit-learn>=1.3.0`
- Data splitting, normalization
- Location: [`src/data_processing/data_split.py`](src/data_processing/data_split.py)

**SciPy** (implied by requirements)
- Signal processing (`scipy.signal`, `scipy.ndimage`)
- Gaussian smoothing for trajectory noise reduction
- Location: [`src/data_processing/feature_engineering_v2.py`](src/data_processing/feature_engineering_v2.py)

---

## Visualization

**Matplotlib 3.7+**
- `matplotlib>=3.7.0`
- Core plotting library
- Radar charts, bar charts, score gauges, comprehensive reports
- Location: [`src/coaching/visualizations.py`](src/coaching/visualizations.py)

**Seaborn 0.12+**
- `seaborn>=0.12.0`
- Statistical visualizations
- Enhanced plot aesthetics

**Plotly 5.14+** (Optional)
- `plotly>=5.14.0`
- Interactive plots (not currently used in production)

---

## Experiment Tracking

**MLflow 2.7+**
- `mlflow>=2.7.0`
- Experiment logging and tracking
- Not actively used in current v2.0 (benchmark-based system)
- Present for future model training workflows

---

## Deployment

**Streamlit**
- Primary deployment method (current)
- Web interface: [`src/deployment/streamlit_app.py`](src/deployment/streamlit_app.py)
- 3-tab layout: Feedback / Visualizations / Priority Actions
- Usage: `streamlit run src/deployment/streamlit_app.py`

**Gradio 4.0+**
- `gradio>=4.0.0`
- Alternative deployment option
- Location: [`src/deployment/coaching_app.py`](src/deployment/coaching_app.py)
- Less actively maintained than Streamlit interface

---

## Development Tools

**Jupyter Ecosystem**
- `jupyter>=1.0.0`, `ipykernel>=6.25.0`, `notebook>=7.0.0`
- Used for exploratory analysis and debugging
- Debug scripts reference notebook-style workflows

**tqdm 4.65+**
- `tqdm>=4.65.0`
- Progress bars for batch processing
- Used in: Pose extraction, feature engineering loops

**Pillow 10.0+**
- `pillow>=10.0.0`
- Image processing utilities

**python-dotenv 1.0+**
- `python-dotenv>=1.0.0`
- Environment variable management (not heavily used)

---

## Testing & Quality (Optional - Not Active)

Currently commented out in `requirements.txt`:
- `pytest>=7.4.0` - Unit testing framework
- `pytest-cov>=4.1.0` - Coverage reporting
- `black>=23.7.0` - Code formatting
- `flake8>=6.1.0` - Linting
- `mypy>=1.5.0` - Type checking

**Status**: No active test suite. System relies on manual testing with diagnostic script ([`diagnose.py`](diagnose.py))

---

## Critical Dependencies & Constraints

**Protobuf 3.20.3** (CRITICAL)
- Not in `requirements.txt` but REQUIRED
- Fixes MediaPipe mutex blocking error
- Must install manually: `pip install protobuf==3.20.3`
- Documented in: [`README.md`](README.md) Troubleshooting section

**Version Lock Chain**:
```
MediaPipe 0.10.9 → TensorFlow < 2.16 → Protobuf 3.20.3
```

Any upgrade to MediaPipe requires validation of entire chain.

---

## Package Management

**requirements.txt**
- Single dependency file
- No virtual environment configuration tracked in repo
- Installation: `pip install -r requirements.txt`
- No `setup.py` or `pyproject.toml` (not packaged as library)

---

## Build & Distribution

**No build process**
- Pure Python, no compilation
- No Docker configuration
- No CI/CD pipelines
- Deployment: Manual environment setup + Streamlit

---

## Data Files & Serialization

**Pickle** (Python stdlib)
- Pose sequences: `data/processed/poses/*.pkl`
- Feature arrays: `data/processed/features/*.pkl`
- Split data: `data/processed/splits/*.pkl` (currently in .gitignore, 105MB file removed from history)

**CSV**
- Metadata: `data/processed/clips/clips_metadata.csv`
- Annotations: `data/annotations/*` (in .gitignore)

---

## Runtime Environment

**Local execution** (primary)
- Development: Jupyter notebooks, Python scripts
- Deployment: Streamlit server on localhost

**Google Colab** (secondary)
- Several debug scripts reference Colab paths
- Example: [`debug_splits_colab.py`](debug_splits_colab.py)
- Used for accessing large dataset files

---

## Notes for Future Development

1. **Testing Infrastructure**: Consider activating pytest suite for regression testing
2. **Type Checking**: MyPy could improve code quality (many untyped functions)
3. **Containerization**: Docker would simplify deployment and dependency management
4. **CI/CD**: GitHub Actions for automated testing on push
5. **Package Structure**: Convert to proper Python package with `setup.py` for easier installation
