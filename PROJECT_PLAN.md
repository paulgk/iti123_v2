# ITI123 Project Implementation Plan
## AI-Based Badminton Stroke Technique Assessment Using Deep Learning

**Student:** Paul
**Project Type:** Model Development Focus
**Dataset:** ShuttleSet (MIT License)
**Duration:** Jan 15 - Feb 26, 2026 (6 weeks)

---

## 📅 Key Deadlines

| Deliverable | Due Date | Status |
|------------|----------|--------|
| Milestone Report | Thu, Jan 29, 2026, 2359 hrs | ⏳ Pending |
| Final Report | Thu, Feb 26, 2026, 2359 hrs | ⏳ Pending |
| Presentation | Wed-Thu, Feb 25-26, 2026 | ⏳ Pending |

---

## 📊 Project Overview

### Problem Statement
Develop an AI-based system that objectively analyzes badminton overhead stroke execution (Clear and Smash) using pose-based deep learning from video footage.

### Technical Approach
1. Extract pose keypoints from video using MediaPipe/MoveNet
2. Engineer biomechanical features from pose sequences
3. Train LSTM/GRU model for temporal sequence classification
4. Deploy as Gradio web application

### Success Metrics
- Minimum: 70% accuracy, complete reports, working demo
- Good: 80% accuracy, thorough analysis, polished interface
- Excellent: 85% accuracy, novel techniques, cloud deployment

---

## 📁 Project Structure

```
iti123_v2/
├── data/
│   ├── raw_videos/           # Original ShuttleSet videos
│   ├── annotations/          # CSV annotation files from ShuttleSet
│   ├── processed/
│   │   ├── clips/           # Extracted stroke clips
│   │   ├── poses/           # Extracted pose sequences
│   │   └── features/        # Engineered feature vectors
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_pose_extraction.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_error_analysis.ipynb
│   └── 06_model_experiments.ipynb
├── src/
│   ├── data_processing/
│   │   ├── extract_clips.py
│   │   ├── extract_poses.py
│   │   ├── player_tracking.py
│   │   ├── feature_engineering.py
│   │   └── data_split.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── lstm_model.py
│   │   └── train.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── visualize.py
│   └── deployment/
│       └── gradio_app.py
├── experiments/              # MLflow tracking
├── models/                   # Saved model checkpoints
├── outputs/                  # Results, plots, reports
├── docs/                     # Documentation
├── app.py                    # Main Gradio application
├── requirements.txt
├── README.md
└── PROJECT_PLAN.md          # This file
```

---

## 🎯 Phase 1: Foundation & Data Pipeline
**Week 1: Jan 15-21, 2026**

### Milestone 1.1: Environment Setup
**Due: Jan 16, 2026**
**Status:** ⬜ Not Started

#### Tasks
- [ ] Create project repository structure (folders above)
- [ ] Set up Python virtual environment
- [ ] Install required libraries:
  ```bash
  pip install tensorflow opencv-python mediapipe pandas numpy matplotlib seaborn scikit-learn mlflow gradio jupyter
  ```
- [ ] Verify installations:
  ```python
  import tensorflow as tf
  import cv2
  import mediapipe as mp
  import gradio as gr
  print("All libraries loaded successfully!")
  ```

#### Deliverable
- ✅ Working development environment
- ✅ All folders created
- ✅ Can import all required libraries

#### Notes
_Add your notes here as you work..._

---

### Milestone 1.2: Data Organization & Exploration
**Due: Jan 18, 2026**
**Status:** ⬜ Not Started

#### Tasks
- [ ] Download ShuttleSet CSV annotations from: https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet
- [ ] Organize video files in `data/raw_videos/` by match ID
- [ ] Parse annotation files:
  - [ ] Read `match.csv`
  - [ ] Read `set1.csv, set2.csv, set3.csv` for each match
- [ ] Filter annotations for Clear (長球) and Smash (殺球, 點扣) only
- [ ] Generate dataset statistics:
  - [ ] Total Clear strokes: ____
  - [ ] Total Smash strokes: ____
  - [ ] Number of matches: ____
  - [ ] Video resolution: ____
  - [ ] Frame rate: ____
  - [ ] Average frames per stroke: ____
- [ ] Create visualizations:
  - [ ] Bar chart: Clear vs Smash distribution
  - [ ] Histogram: Stroke duration distribution
  - [ ] Table: Top 10 matches by stroke count
- [ ] Document in `notebooks/01_data_exploration.ipynb`

#### Deliverable
- ✅ Organized dataset with documented statistics
- ✅ Jupyter notebook with visualizations
- ✅ Written summary of findings

#### Dataset Statistics (Fill in after completion)
```
Total strokes analyzed: _____
Clear strokes: _____
Smash strokes: _____
Wrist smash strokes: _____
Total matches: _____
Video resolution: _____
Frame rate: _____ fps
Average stroke duration: _____ frames (_____ seconds)
```

#### Notes
_Add your notes here..._

---

### Milestone 1.3: Video Clip Extraction
**Due: Jan 21, 2026**
**Status:** ⬜ Not Started

#### Tasks
- [ ] Implement `src/data_processing/extract_clips.py`:
  ```python
  # Pseudocode structure:
  def extract_clip(video_path, frame_start, frame_end, output_path):
      # Load video
      # Extract frames from frame_start to frame_end
      # Add temporal context (±30 frames / ±1 sec)
      # Save as new video clip
      pass
  ```
- [ ] Design naming convention: `{match_id}_{rally_id}_{stroke_id}_{stroke_type}.mp4`
- [ ] Handle edge cases:
  - [ ] Clips at start of video (frame_start < 30)
  - [ ] Clips at end of video (frame_end > total_frames - 30)
  - [ ] Invalid frame numbers in annotations
- [ ] Process all Clear and Smash strokes
- [ ] Validate 20 random clips manually (watch them)
- [ ] Create metadata CSV linking clips to annotations
- [ ] Generate extraction report

#### Deliverable
- ✅ Extracted clips in `data/processed/clips/`
- ✅ Metadata file: `data/processed/clips/metadata.csv`
- ✅ Extraction report with statistics
- ✅ Script: `src/data_processing/extract_clips.py`

#### Extraction Report (Fill in after completion)
```
Clips Extracted Successfully:
- Clear clips: _____
- Smash clips: _____
- Wrist smash clips: _____
- Total: _____

Failed Extractions: _____
Reasons for failures:
- Invalid frame numbers: _____
- Video file not found: _____
- Other: _____

Average clip duration: _____ seconds
Clips validated manually: 20/20 ✅
```

#### Notes
_Add your notes here..._

---

## 🎯 Phase 2: Pose Estimation Pipeline
**Week 2: Jan 22-28, 2026**

### Milestone 2.1: Pose Estimation Implementation
**Due: Jan 24, 2026**
**Status:** ⬜ Not Started

#### Tasks
- [ ] Choose pose estimation model:
  - Option A: ☐ MediaPipe Pose (easier, faster, 33 keypoints)
  - Option B: ☐ MoveNet Thunder (more accurate, 17 keypoints)
  - **Selected:** _________

- [ ] Implement `src/data_processing/extract_poses.py`:
  ```python
  # Pseudocode structure:
  def extract_poses_from_video(video_path):
      # Initialize pose model
      # For each frame:
          # Detect pose
          # Extract keypoints (x, y, confidence)
          # Store in sequence
      # Return pose_sequence: (num_frames, num_keypoints, 3)
      pass
  ```

- [ ] Implement quality checks:
  - [ ] Filter keypoints with confidence < 0.5
  - [ ] Detect missing keypoints (confidence = 0)
  - [ ] Flag clips with >50% missing keypoints

- [ ] Process 100 sample clips for testing
- [ ] Create visualization function:
  ```python
  def visualize_pose_overlay(video_path, pose_sequence, output_path):
      # Draw skeleton overlay on video frames
      # Save as new video
      pass
  ```
- [ ] Generate visualization for 10 sample clips
- [ ] Document in `notebooks/02_pose_extraction.ipynb`

#### Deliverable
- ✅ Script: `src/data_processing/extract_poses.py`
- ✅ Notebook: `notebooks/02_pose_extraction.ipynb`
- ✅ 10 visualization videos with pose overlays
- ✅ Quality metrics report

#### Quality Metrics (Fill in after completion)
```
Model Used: _________
Number of keypoints: _____

Test Set: 100 clips
Average confidence score: _____
Clips with >90% keypoint detection: _____ (____%)
Clips with >80% keypoint detection: _____ (____%)
Clips with <50% keypoint detection: _____ (____%)

Average processing time per clip: _____ seconds
```

#### Notes
_Add your notes here..._

---

### Milestone 2.2: Player Tracking for Multi-Player Scenes
**Due: Jan 26, 2026**
**Status:** ⬜ Not Started

#### Tasks
- [ ] Analyze multi-player scenarios:
  - [ ] How many clips have multiple players visible? _____
  - [ ] Are most clips singles or doubles? _________

- [ ] Implement player identification strategy:
  - Option A: ☐ Use court position + movement continuity
  - Option B: ☐ Use bounding box detection (YOLO) + tracking
  - Option C: ☐ Manual filtering (focus on singles only)
  - **Selected:** _________

- [ ] If implementing automated tracking:
  - [ ] Implement `src/data_processing/player_tracking.py`
  - [ ] Detect multiple players per frame
  - [ ] Identify target player (stroke executor)
  - [ ] Track player across frames
  - [ ] Handle occlusions

- [ ] Validate tracking on 50 clips:
  - [ ] 25 singles match clips
  - [ ] 25 doubles match clips

- [ ] Calculate tracking accuracy
- [ ] Filter pose sequences to target player only

#### Deliverable
- ✅ Script: `src/data_processing/player_tracking.py` (if automated)
- ✅ Tracking validation report
- ✅ Filtered pose sequences

#### Tracking Accuracy (Fill in after completion)
```
Approach used: _________

Singles matches:
- Successful tracking: _____/25 (____%)
- Failed cases: _____

Doubles matches:
- Successful tracking: _____/25 (____%)
- Failed cases: _____

Decision:
☐ Proceed with automated tracking
☐ Focus on singles matches only
☐ Manual filtering for doubles
```

#### Notes
_Add your notes here..._

---

### Milestone 2.3: Feature Engineering
**Due: Jan 28, 2026**
**Status:** ⬜ Not Started

#### Tasks
- [ ] Design biomechanical features:

  **A. Joint Angles**
  - [ ] Shoulder angle (shoulder-elbow-wrist)
  - [ ] Elbow angle
  - [ ] Hip-shoulder angle (torso rotation)

  **B. Velocities**
  - [ ] Angular velocity of arm swing
  - [ ] Wrist velocity (racket speed proxy)
  - [ ] Center of mass velocity

  **C. Spatial Features**
  - [ ] Arm extension (shoulder-to-wrist distance)
  - [ ] Body height (normalized)
  - [ ] Stance width

  **D. Temporal Features**
  - [ ] Time to peak velocity
  - [ ] Acceleration phase duration
  - [ ] Deceleration phase duration

- [ ] Implement `src/data_processing/feature_engineering.py`:
  ```python
  def engineer_features(pose_sequence):
      # For each frame:
          # Calculate joint angles
          # Calculate velocities (frame-to-frame differences)
          # Calculate spatial measurements
      # Normalize by player height
      # Return feature_sequence: (num_frames, num_features)
      pass
  ```

- [ ] Handle missing keypoints:
  - [ ] Linear interpolation for short gaps (1-3 frames)
  - [ ] Flag sequences with long gaps (>5 frames)

- [ ] Normalize features:
  - [ ] By player body proportions (height, arm length)
  - [ ] Temporal alignment (align by impact frame)

- [ ] Visualize features:
  - [ ] Plot feature distributions (Clear vs Smash)
  - [ ] Create correlation heatmap
  - [ ] Generate box plots for key features

- [ ] Document in `notebooks/03_feature_engineering.ipynb`

#### Deliverable
- ✅ Script: `src/data_processing/feature_engineering.py`
- ✅ Notebook: `notebooks/03_feature_engineering.ipynb`
- ✅ Feature vectors saved for all clips
- ✅ Feature analysis visualizations

#### Feature Summary (Fill in after completion)
```
Total features engineered: _____

Feature categories:
- Joint angles: _____
- Velocities: _____
- Spatial measurements: _____
- Temporal features: _____

Feature vector shape per clip: (_____ frames, _____ features)

Key findings from visualization:
- Most discriminative features for Clear vs Smash:
  1. _________
  2. _________
  3. _________
```

#### Notes
_Add your notes here..._

---

## 🎯 Phase 3: Baseline Model & Milestone Report
**Week 3: Jan 29, 2026**

### Milestone 3.1: Data Preparation
**Due: Jan 29, 2026 (Morning)**
**Status:** ⬜ Not Started

#### Tasks
- [ ] Implement `src/data_processing/data_split.py`
- [ ] Split data:
  - [ ] Training: 70%
  - [ ] Validation: 15%
  - [ ] Test: 15%
- [ ] Ensure stratification (balanced Clear/Smash ratio in each set)
- [ ] Split by match (no clips from same match in different sets)
- [ ] Verify no data leakage
- [ ] Create data loaders for training

#### Deliverable
- ✅ Script: `src/data_processing/data_split.py`
- ✅ Split files saved
- ✅ Split statistics documented

#### Split Statistics (Fill in after completion)
```
Training Set:
- Total clips: _____
- Clear: _____ (____%)
- Smash: _____ (____%)
- Matches: _____

Validation Set:
- Total clips: _____
- Clear: _____ (____%)
- Smash: _____ (____%)
- Matches: _____

Test Set (held out until final evaluation):
- Total clips: _____
- Clear: _____ (____%)
- Smash: _____ (____%)
- Matches: _____
```

#### Notes
_Add your notes here..._

---

### Milestone 3.2: Baseline Model
**Due: Jan 29, 2026 (Afternoon)**
**Status:** ⬜ Not Started

#### Tasks
- [ ] Choose baseline approach:
  - Option A: ☐ Random Forest on aggregated features (mean, max, std)
  - Option B: ☐ Logistic Regression on flattened features
  - Option C: ☐ Simple MLP (Multi-Layer Perceptron)
  - **Selected:** _________

- [ ] Implement `src/models/baseline.py`
- [ ] Train baseline model on training set
- [ ] Evaluate on validation set
- [ ] Generate performance metrics:
  - [ ] Accuracy
  - [ ] Precision (per class)
  - [ ] Recall (per class)
  - [ ] F1-score (per class)
  - [ ] Confusion matrix
- [ ] Analyze errors
- [ ] Document in `notebooks/04_baseline_model.ipynb`

#### Deliverable
- ✅ Script: `src/models/baseline.py`
- ✅ Notebook: `notebooks/04_baseline_model.ipynb`
- ✅ Saved baseline model
- ✅ Results table and confusion matrix

#### Baseline Results (Fill in after completion)
```
Model: _________

Training Set Performance:
- Accuracy: _____%

Validation Set Performance:
- Accuracy: _____%

Class-wise Metrics (Validation):
Clear:
  - Precision: _____
  - Recall: _____
  - F1-score: _____

Smash:
  - Precision: _____
  - Recall: _____
  - F1-score: _____

Confusion Matrix:
              Predicted
              Clear  Smash
Actual Clear  ____   ____
       Smash  ____   ____
```

#### Notes
_Add your notes here..._

---

### Milestone 3.3: MILESTONE REPORT SUBMISSION
**Due: Jan 29, 2026, 2359 hrs** ⚠️
**Status:** ⬜ Not Started

#### Required Sections (8 pages max, excluding references)

1. **Introduction & Motivation** (1 page)
   - [ ] Problem statement
   - [ ] Business motivation (why this matters)
   - [ ] User needs addressed
   - [ ] Project objectives

2. **Dataset** (1.5 pages)
   - [ ] ShuttleSet description
   - [ ] Dataset statistics (with charts)
   - [ ] Data preprocessing steps
   - [ ] Sample visualizations
   - [ ] Citation: Wang et al., KDD 2023

3. **Methodology** (2.5 pages)
   - [ ] Pose estimation approach (MediaPipe/MoveNet)
   - [ ] Feature engineering details
   - [ ] Baseline model architecture
   - [ ] Planned LSTM/GRU architecture (with diagram)
   - [ ] Training procedure

4. **Preliminary Experiments** (2 pages)
   - [ ] Baseline results (tables & charts)
   - [ ] Error analysis
   - [ ] Sample predictions (successful and failed)
   - [ ] Comparison discussion

5. **Responsible AI** (0.5 page)
   - [ ] Potential biases (dataset focused on professional players)
   - [ ] Limitations (may not generalize to recreational players)
   - [ ] Mitigation strategies (pose normalization, clear documentation)
   - [ ] Ethical considerations

6. **Next Steps** (0.5 page)
   - [ ] LSTM/GRU implementation plan
   - [ ] Hyperparameter tuning strategy
   - [ ] Deployment plan (Gradio)
   - [ ] Expected improvements

#### Deliverable
- ✅ PDF report submitted to portal before 2359 hrs
- ✅ Report follows formatting guidelines
- ✅ All figures properly labeled
- ✅ References cited correctly

#### Submission Checklist
- [ ] Report is 8 pages or less (excluding references)
- [ ] All sections complete
- [ ] Figures have captions
- [ ] Tables are formatted properly
- [ ] ShuttleSet dataset properly cited
- [ ] Grammar and spelling checked
- [ ] PDF file named correctly: `ITI123_Milestone_YourName.pdf`
- [ ] Submitted to correct portal before deadline

#### Notes
_Add your notes here..._

---

## 🎯 Phase 4: Deep Learning Model Development
**Week 4: Jan 30 - Feb 5, 2026**

### Milestone 4.1: LSTM/GRU Implementation
**Due: Feb 1, 2026**
**Status:** ⬜ Not Started

#### Tasks
- [ ] Design model architecture:
  ```
  Proposed Architecture:
  Input: (batch_size, sequence_length, num_features)
    ↓
  Bidirectional LSTM (128 units)
    ↓
  Dropout (0.3)
    ↓
  Bidirectional LSTM (64 units)
    ↓
  Dropout (0.3)
    ↓
  Dense (64 units) + ReLU
    ↓
  Dropout (0.3)
    ↓
  Dense (2 units) + Softmax
    ↓
  Output: [Clear probability, Smash probability]
  ```

- [ ] Implement model in `src/models/lstm_model.py`:
  ```python
  def build_lstm_model(sequence_length, num_features):
      # Build model architecture
      # Compile with appropriate loss and optimizer
      return model
  ```

- [ ] Set up training configuration:
  - [ ] Loss function: Categorical Crossentropy
  - [ ] Optimizer: Adam (learning_rate=0.001)
  - [ ] Metrics: Accuracy, AUC
  - [ ] Batch size: 32
  - [ ] Initial epochs: 50

- [ ] Implement data augmentation:
  - [ ] Temporal jittering (shift sequences by ±5 frames)
  - [ ] Add Gaussian noise to features (std=0.01)
  - [ ] Random temporal crop

- [ ] Set up MLflow experiment tracking:
  - [ ] Log hyperparameters
  - [ ] Log metrics (loss, accuracy) per epoch
  - [ ] Save model checkpoints

- [ ] Implement training script `src/models/train.py`

#### Deliverable
- ✅ Script: `src/models/lstm_model.py`
- ✅ Script: `src/models/train.py`
- ✅ Model architecture diagram
- ✅ MLflow tracking configured

#### Model Configuration (Fill in)
```
Architecture Details:
- Sequence length: _____ frames
- Number of features: _____
- LSTM layer 1 units: _____
- LSTM layer 2 units: _____
- Dropout rate: _____
- Dense layer units: _____
- Output classes: 2 (Clear, Smash)

Training Configuration:
- Loss: Categorical Crossentropy
- Optimizer: Adam
- Learning rate: _____
- Batch size: _____
- Epochs: _____
```

#### Notes
_Add your notes here..._

---

### Milestone 4.2: Initial Training
**Due: Feb 3, 2026**
**Status:** ⬜ Not Started

#### Tasks
- [ ] Train LSTM model on training set
- [ ] Monitor training progress:
  - [ ] Plot training loss curve
  - [ ] Plot validation loss curve
  - [ ] Plot training accuracy curve
  - [ ] Plot validation accuracy curve
- [ ] Implement early stopping:
  - [ ] Patience: 10 epochs
  - [ ] Monitor: validation loss
- [ ] Save best model checkpoint (lowest validation loss)
- [ ] Evaluate best model on validation set
- [ ] Compare with baseline model
- [ ] Log all results in MLflow
- [ ] Document training process

#### Deliverable
- ✅ Trained model: `models/lstm_v1.h5` or `models/lstm_v1.pth`
- ✅ Training curves (plots)
- ✅ Validation results
- ✅ Comparison table with baseline
- ✅ MLflow experiment logged

#### Training Results (Fill in after completion)
```
Training Summary:
- Total epochs trained: _____
- Best epoch: _____
- Training time: _____ minutes
- Final training loss: _____
- Final validation loss: _____
- Early stopping triggered: ☐ Yes ☐ No

Validation Set Performance:
- Accuracy: _____%

Class-wise Metrics:
Clear:
  - Precision: _____
  - Recall: _____
  - F1-score: _____

Smash:
  - Precision: _____
  - Recall: _____
  - F1-score: _____

Comparison with Baseline:
- Baseline accuracy: _____%
- LSTM accuracy: _____%
- Improvement: +_____%
```

#### Notes
_Add your notes here..._

---

### Milestone 4.3: Error Analysis
**Due: Feb 5, 2026**
**Status:** ⬜ Not Started

#### Tasks
- [ ] Identify all misclassified examples from validation set:
  - [ ] False positives (predicted Clear, actually Smash): _____
  - [ ] False negatives (predicted Smash, actually Clear): _____

- [ ] Analyze failure patterns:
  - [ ] Are failures correlated with specific players? _____
  - [ ] Are failures correlated with specific matches? _____
  - [ ] Are failures correlated with pose estimation quality? _____
  - [ ] Are there ambiguous strokes (borderline cases)? _____

- [ ] Visualize confused samples:
  - [ ] Select 10 worst false positives
  - [ ] Select 10 worst false negatives
  - [ ] Create visualization with:
    - Original video
    - Pose overlay
    - Feature plots
    - Model prediction
    - Ground truth label

- [ ] Conduct statistical analysis:
  - [ ] Error rate by confidence score
  - [ ] Error rate by stroke duration
  - [ ] Error rate by player

- [ ] Generate insights and recommendations:
  - [ ] What can be improved in pose estimation?
  - [ ] What features might be missing?
  - [ ] What model changes might help?

- [ ] Document in `notebooks/05_error_analysis.ipynb`

#### Deliverable
- ✅ Notebook: `notebooks/05_error_analysis.ipynb`
- ✅ Confusion matrix heatmap
- ✅ Error analysis report with visualizations
- ✅ Recommendations for improvement

#### Error Analysis Summary (Fill in)
```
Misclassification Breakdown:
- False Positives (Clear→Smash): _____ (____%)
- False Negatives (Smash→Clear): _____ (____%)

Failure Patterns Identified:
1. _________
2. _________
3. _________

Key Insights:
1. _________
2. _________
3. _________

Recommended Improvements:
1. _________
2. _________
3. _________
```

#### Notes
_Add your notes here..._

---

## 🎯 Phase 5: Optimization & Enhancement
**Week 5: Feb 6-12, 2026**

### Milestone 5.1: Hyperparameter Tuning
**Due: Feb 8, 2026**
**Status:** ⬜ Not Started

#### Tasks
- [ ] Define hyperparameter search space:

  **A. Architecture Parameters**
  - [ ] LSTM units layer 1: [64, 128, 256]
  - [ ] LSTM units layer 2: [32, 64, 128]
  - [ ] Number of LSTM layers: [1, 2, 3]
  - [ ] Dropout rate: [0.2, 0.3, 0.4, 0.5]
  - [ ] Dense layer units: [32, 64, 128]

  **B. Training Parameters**
  - [ ] Learning rate: [0.0001, 0.001, 0.01]
  - [ ] Batch size: [16, 32, 64]
  - [ ] Sequence length: [30, 50, 75, 100] frames

  **C. Data Parameters**
  - [ ] Augmentation strength: [low, medium, high]

- [ ] Implement systematic search:
  - Option A: ☐ Grid search (exhaustive)
  - Option B: ☐ Random search (sample N configs)
  - Option C: ☐ Manual iterative search
  - **Selected:** _________

- [ ] Run experiments:
  - [ ] Track all experiments in MLflow
  - [ ] Use validation set for evaluation
  - [ ] Train at least 10 different configurations

- [ ] Select best configuration based on:
  - [ ] Validation accuracy (primary)
  - [ ] Validation loss
  - [ ] Training stability
  - [ ] Inference speed

- [ ] Retrain best model with more epochs if needed

#### Deliverable
- ✅ Hyperparameter search results table
- ✅ Best configuration identified
- ✅ Retrained optimized model
- ✅ MLflow experiments logged
- ✅ Notebook: `notebooks/06_model_experiments.ipynb`

#### Hyperparameter Search Results (Fill in)
```
Configurations Tested: _____

Top 5 Configurations:

1. Config #_____
   - LSTM units: [_____, _____]
   - Dropout: _____
   - Learning rate: _____
   - Batch size: _____
   - Validation accuracy: _____%

2. Config #_____
   - LSTM units: [_____, _____]
   - Dropout: _____
   - Learning rate: _____
   - Batch size: _____
   - Validation accuracy: _____%

3. Config #_____
   - LSTM units: [_____, _____]
   - Dropout: _____
   - Learning rate: _____
   - Batch size: _____
   - Validation accuracy: _____%

[Continue for top 5...]

Selected Best Configuration: Config #_____
Reason for selection: _________
```

#### Notes
_Add your notes here..._

---

### Milestone 5.2: Advanced Techniques (Optional)
**Due: Feb 10, 2026**
**Status:** ⬜ Not Started

#### Tasks
Choose 1-2 advanced techniques to implement:

**Option A: Attention Mechanism**
- [ ] Implement temporal attention layer
- [ ] Visualize attention weights
- [ ] Compare with baseline LSTM

**Option B: Multi-Task Learning**
- [ ] Add auxiliary task (e.g., predict stroke quality score)
- [ ] Implement multi-task loss
- [ ] Evaluate on both tasks

**Option C: Alternative Architecture**
- [ ] Try GRU instead of LSTM
- [ ] Try CNN-LSTM hybrid (1D CNN + LSTM)
- [ ] Try Transformer encoder
- [ ] Compare with original LSTM

**Option D: Ensemble Methods**
- [ ] Train 3-5 models with different seeds
- [ ] Implement ensemble voting/averaging
- [ ] Evaluate ensemble performance

**Selected Techniques:**
- [ ] Technique 1: _________
- [ ] Technique 2: _________

#### Deliverable
- ✅ Implementation of chosen technique(s)
- ✅ Comparison results
- ✅ Analysis of what worked / didn't work

#### Advanced Techniques Results (Fill in)
```
Technique 1: _________
- Implementation status: ☐ Complete ☐ Partial ☐ Abandoned
- Performance: _____%
- Comparison with baseline: +/- _____%
- Key findings: _________

Technique 2: _________
- Implementation status: ☐ Complete ☐ Partial ☐ Abandoned
- Performance: _____%
- Comparison with baseline: +/- _____%
- Key findings: _________
```

#### Notes
_Add your notes here..._

---

### Milestone 5.3: Final Model Selection
**Due: Feb 12, 2026**
**Status:** ⬜ Not Started

#### Tasks
- [ ] Compare all model variants on validation set:
  - [ ] Baseline model
  - [ ] LSTM v1 (initial)
  - [ ] LSTM optimized (best hyperparameters)
  - [ ] Advanced technique variants (if implemented)

- [ ] Select final model based on:
  - [ ] Validation accuracy
  - [ ] Robustness (consistent performance)
  - [ ] Inference speed
  - [ ] Model size

- [ ] Retrain final model on combined train+validation data
  - **Note:** Only do this after finalizing model choice!

- [ ] Evaluate on held-out test set (first time!)
  - [ ] Accuracy
  - [ ] Precision, Recall, F1 per class
  - [ ] Confusion matrix
  - [ ] ROC curve and AUC

- [ ] Generate comprehensive results report:
  - [ ] Performance tables
  - [ ] Visualizations (curves, matrices)
  - [ ] Error analysis on test set
  - [ ] Statistical significance tests (if applicable)

- [ ] Save final model with documentation

#### Deliverable
- ✅ Final model saved: `models/final_model.h5`
- ✅ Complete test set results
- ✅ Model comparison table
- ✅ Final model documentation

#### Final Model Selection (Fill in)
```
Models Compared:
1. Baseline: _____%
2. LSTM v1: _____%
3. LSTM optimized: _____%
4. [Advanced technique]: _____%

Selected Final Model: _________
Reason: _________

Final Model Architecture:
- Input shape: (_____, _____)
- LSTM layers: _____
- Parameters: _____
- Model size: _____ MB
- Inference time per clip: _____ ms
```

#### Final Test Set Results (Fill in)
```
Test Set Performance:
- Accuracy: _____%
- Macro F1-score: _____

Class-wise Metrics:
Clear:
  - Precision: _____
  - Recall: _____
  - F1-score: _____
  - Support: _____

Smash:
  - Precision: _____
  - Recall: _____
  - F1-score: _____
  - Support: _____

Confusion Matrix:
              Predicted
              Clear  Smash
Actual Clear  ____   ____
       Smash  ____   ____

ROC AUC Score: _____
```

#### Model Interpretability Insights
```
Most important features (if analyzable):
1. _________
2. _________
3. _________

Typical failure cases:
1. _________
2. _________
```

#### Notes
_Add your notes here..._

---

## 🎯 Phase 6: Deployment & Documentation
**Week 6: Feb 13-25, 2026**

### Milestone 6.1: Gradio Interface Development
**Due: Feb 16, 2026**
**Status:** ⬜ Not Started

#### Tasks
- [ ] Design Gradio interface layout:
  ```
  [Title: Badminton Stroke Analyzer]

  Input Section:
  - Video file upload (.mp4, .avi)
  - [Analyze Button]

  Output Section:
  - Processed video with pose overlay
  - Prediction: Clear / Smash
  - Confidence: ____%
  - Technique metrics:
    - Arm extension: _____
    - Max swing velocity: _____
    - Impact timing: _____
  - Performance chart (feature visualization)
  ```

- [ ] Implement `src/deployment/gradio_app.py`:
  ```python
  import gradio as gr

  def analyze_stroke(video_file):
      # 1. Extract poses from video
      poses = extract_poses(video_file)

      # 2. Engineer features
      features = engineer_features(poses)

      # 3. Predict with model
      prediction, confidence = model.predict(features)

      # 4. Calculate technique metrics
      metrics = calculate_metrics(features, poses)

      # 5. Generate visualization
      viz_video = create_visualization(video_file, poses, prediction)

      return viz_video, prediction, confidence, metrics
  ```

- [ ] Implement end-to-end pipeline
- [ ] Add error handling:
  - [ ] Invalid video format
  - [ ] Pose detection failure
  - [ ] Model inference errors

- [ ] Optimize for speed:
  - [ ] Target: <10 seconds per 2-3 second clip
  - [ ] Use GPU if available
  - [ ] Consider model quantization

- [ ] Test with various inputs:
  - [ ] Sample Clear strokes (10 clips)
  - [ ] Sample Smash strokes (10 clips)
  - [ ] Edge cases (poor lighting, occlusion)

- [ ] Create demo video (2-3 minutes) showing:
  - [ ] Upload video
  - [ ] Processing
  - [ ] Results display

- [ ] Write usage instructions

#### Deliverable
- ✅ Main app script: `app.py`
- ✅ Working local demo
- ✅ Screenshots of interface (3-5 images)
- ✅ Demo video (2-3 minutes)
- ✅ Usage instructions

#### Interface Testing Results (Fill in)
```
Test Cases:
1. Clear stroke (good quality): ☐ Pass ☐ Fail
2. Smash stroke (good quality): ☐ Pass ☐ Fail
3. Poor lighting: ☐ Pass ☐ Fail
4. Partial occlusion: ☐ Pass ☐ Fail
5. Multiple players: ☐ Pass ☐ Fail

Performance:
- Average processing time: _____ seconds
- GPU utilized: ☐ Yes ☐ No
- Memory usage: _____ MB

User Experience:
- Interface clarity: ☐ Excellent ☐ Good ☐ Needs improvement
- Error messages helpful: ☐ Yes ☐ No
- Results easy to understand: ☐ Yes ☐ No
```

#### Notes
_Add your notes here..._

---

### Milestone 6.2: Cloud Deployment (Optional)
**Due: Feb 18, 2026**
**Status:** ⬜ Not Started

#### Tasks (Optional - do if time permits)
- [ ] Choose deployment platform:
  - Option A: ☐ Hugging Face Spaces
  - Option B: ☐ Streamlit Cloud
  - Option C: ☐ AWS/Google Cloud
  - **Selected:** _________

- [ ] Prepare for deployment:
  - [ ] Create requirements.txt
  - [ ] Optimize model size (quantization)
  - [ ] Add deployment configurations

- [ ] Deploy application
- [ ] Test public access
- [ ] Share demo link

#### Deliverable (if completed)
- ✅ Public URL: _________
- ✅ Deployment documentation

#### Notes
_Add your notes here..._

---

### Milestone 6.3: Code Organization & Documentation
**Due: Feb 20, 2026**
**Status:** ⬜ Not Started

#### Tasks
- [ ] Code cleanup and refactoring:
  - [ ] Remove unused code
  - [ ] Consistent naming conventions
  - [ ] Proper code formatting (PEP 8)
  - [ ] Add type hints

- [ ] Add comprehensive docstrings:
  - [ ] All functions documented
  - [ ] All classes documented
  - [ ] Include parameter descriptions
  - [ ] Include return value descriptions
  - [ ] Add usage examples

- [ ] Create `README.md`:
  ```markdown
  # AI-Based Badminton Stroke Assessment

  ## Overview
  [Brief description]

  ## Installation
  [Step-by-step setup]

  ## Usage
  [How to run the code]

  ## Dataset
  [ShuttleSet citation and usage]

  ## Model Architecture
  [Architecture diagram and description]

  ## Results
  [Summary of performance]

  ## Demo
  [Link to Gradio app or demo video]

  ## Citation
  [How to cite this work]
  ```

- [ ] Create `requirements.txt`:
  ```
  tensorflow==2.x.x
  opencv-python==4.x.x
  mediapipe==0.x.x
  pandas==1.x.x
  numpy==1.x.x
  matplotlib==3.x.x
  seaborn==0.x.x
  scikit-learn==1.x.x
  mlflow==2.x.x
  gradio==4.x.x
  jupyter==1.x.x
  ```

- [ ] Create file structure documentation:
  - [ ] Explain what each script does
  - [ ] List key functions
  - [ ] Data flow diagram

- [ ] Upload to GitHub (or prepare ZIP):
  - [ ] Create .gitignore (exclude videos, large files)
  - [ ] Push all code
  - [ ] Verify repository is accessible

#### Deliverable
- ✅ Clean, documented codebase
- ✅ Comprehensive README.md
- ✅ requirements.txt
- ✅ File structure documentation
- ✅ GitHub repository (or prepared ZIP <5MB)

#### Documentation Checklist
- [ ] README.md is comprehensive and clear
- [ ] All major functions have docstrings
- [ ] Installation instructions tested on fresh environment
- [ ] Usage examples are clear
- [ ] Dataset properly cited
- [ ] File structure is well-organized
- [ ] No hardcoded paths (use relative paths or configs)

#### Repository Information (Fill in)
```
GitHub Repository: _________
or
ZIP file prepared: ☐ Yes (Size: _____ MB)

Key Files:
- Main app: app.py
- Training script: src/models/train.py
- Data processing: src/data_processing/
- Notebooks: notebooks/
- Documentation: README.md
```

#### Notes
_Add your notes here..._

---

### Milestone 6.4: Final Report Writing
**Due: Feb 24, 2026**
**Status:** ⬜ Not Started

#### Report Structure (15 pages max, excluding references)

**1. Introduction & Motivation** (1.5 pages)
- [ ] Problem statement (clear and specific)
- [ ] Business motivation (why this matters to users)
- [ ] User needs addressed
- [ ] Project objectives
- [ ] Contribution and novelty

**2. Related Work** (1 page)
- [ ] Brief literature review on sports analysis AI
- [ ] Pose estimation methods
- [ ] Badminton-specific AI systems
- [ ] Gap that this project addresses

**3. Dataset** (2 pages)
- [ ] ShuttleSet detailed description
- [ ] Dataset statistics (tables and charts)
- [ ] Data preprocessing pipeline (with flowchart)
- [ ] Train/validation/test split
- [ ] Data quality and limitations
- [ ] Sample visualizations
- [ ] Proper citation: Wang et al., KDD 2023

**4. Methodology** (3.5 pages)
- [ ] System overview (architecture diagram)
- [ ] Pose estimation approach
  - Model selection (MediaPipe/MoveNet)
  - Player tracking (if applicable)
  - Quality assurance
- [ ] Feature engineering
  - Biomechanical features designed
  - Normalization strategies
  - Temporal alignment
- [ ] Model architecture
  - LSTM/GRU design (detailed diagram)
  - Hyperparameters
  - Training procedure
  - Loss function and optimizer
- [ ] Data augmentation techniques
- [ ] Experiment tracking methodology

**5. Experiments & Results** (4 pages)
- [ ] Experimental setup
- [ ] Baseline model results
  - Performance metrics
  - Analysis
- [ ] Deep learning model results
  - Initial LSTM results
  - Hyperparameter tuning results
  - Advanced technique results (if applicable)
- [ ] Model comparison table (all variants)
- [ ] Learning curves (training/validation)
- [ ] Final test set results
  - Accuracy, Precision, Recall, F1
  - Confusion matrix (heatmap)
  - ROC curve and AUC
  - Class-wise performance
- [ ] Error analysis
  - Failure case analysis
  - Sample visualizations
  - Statistical significance tests
- [ ] Ablation studies (if applicable)
  - Impact of different features
  - Impact of architecture choices

**6. Model Interpretability** (1 page)
- [ ] Feature importance analysis
- [ ] Attention visualization (if applicable)
- [ ] What the model learned
- [ ] Validation against domain knowledge

**7. Deployment** (1.5 pages)
- [ ] System architecture
- [ ] Gradio interface design
- [ ] End-to-end pipeline
- [ ] User interaction flow
- [ ] Screenshots of interface
- [ ] Performance optimization
- [ ] Deployment considerations (cloud vs local)
- [ ] Demo link (if available)

**8. Responsible AI** (1.5 pages)
- [ ] Potential biases identified
  - Dataset bias (professional players only)
  - Pose estimation bias
  - Model bias
- [ ] Limitations clearly stated
  - Generalization limitations
  - Technical limitations
  - Use case limitations
- [ ] Mitigation strategies implemented
  - Normalization
  - Quality checks
  - Clear documentation
- [ ] Ethical considerations
  - Privacy (no PII)
  - Appropriate use cases
  - Misuse prevention

**9. Conclusion & Future Work** (1 page)
- [ ] Summary of achievements
- [ ] Key contributions
- [ ] Limitations acknowledged
- [ ] Future improvements
  - More stroke types
  - Real-time processing
  - Multi-player analysis
  - Technique correction suggestions
- [ ] Broader impact

**10. References** (separate page)
- [ ] ShuttleSet paper (Wang et al., KDD 2023)
- [ ] Pose estimation papers (MediaPipe/MoveNet)
- [ ] LSTM/GRU papers (Hochreiter & Schmidhuber, etc.)
- [ ] Related sports AI papers
- [ ] Framework/library documentation
- [ ] All citations formatted consistently

#### Writing Schedule
- [ ] Feb 21: Draft sections 1-3 (Introduction, Related Work, Dataset)
- [ ] Feb 22: Draft sections 4-5 (Methodology, Experiments)
- [ ] Feb 23: Draft sections 6-9 (Interpretability, Deployment, Responsible AI, Conclusion)
- [ ] Feb 24 Morning: Polish all sections, add visualizations
- [ ] Feb 24 Afternoon: Proofread, format, finalize

#### Deliverable
- ✅ Complete Final Report (PDF)
- ✅ All figures properly labeled
- ✅ All tables formatted
- ✅ References complete
- ✅ Grammar and spelling checked

#### Final Report Checklist
- [ ] Page limit: 15 pages or less (excluding references)
- [ ] All required sections included
- [ ] Figures have captions and are referenced in text
- [ ] Tables are clear and properly formatted
- [ ] ShuttleSet properly cited
- [ ] All claims supported by evidence (results, citations)
- [ ] Writing is clear and concise
- [ ] Technical terms explained
- [ ] No grammatical errors or typos
- [ ] PDF file named correctly: `ITI123_Final_YourName.pdf`
- [ ] Consistent formatting throughout

#### Notes
_Add your notes here..._

---

### Milestone 6.5: Presentation Preparation
**Due: Feb 25, 2026**
**Status:** ⬜ Not Started

#### Tasks
- [ ] Create presentation slides (10-15 slides):

  **Slide Structure:**
  1. Title Slide
     - Project title
     - Your name
     - Date

  2. Problem & Motivation (1 slide)
     - What problem are you solving?
     - Why does it matter?

  3. Dataset Overview (1 slide)
     - ShuttleSet description
     - Key statistics
     - Sample images

  4. Methodology Overview (2-3 slides)
     - System architecture diagram
     - Pose estimation → Features → Model → Output
     - Key technical choices

  5. Model Architecture (1 slide)
     - LSTM/GRU diagram
     - Input/output shapes
     - Key components

  6. Results (2-3 slides)
     - Performance comparison table
     - Confusion matrix
     - Learning curves
     - Key metrics highlighted

  7. Demo (1-2 slides)
     - Live demo OR
     - Demo video (30-60 seconds)
     - Gradio interface screenshots

  8. Error Analysis & Insights (1 slide)
     - What worked well
     - What didn't work
     - Key learnings

  9. Challenges & Solutions (1 slide)
     - Technical challenges faced
     - How you solved them

  10. Responsible AI (1 slide)
      - Limitations
      - Ethical considerations
      - Appropriate use

  11. Conclusion & Future Work (1 slide)
      - Summary of achievements
      - Future improvements

  12. Thank You / Questions (1 slide)

- [ ] Prepare demo:
  - [ ] Option A: Live demo with Gradio app
    - [ ] Test video clips ready (2-3 clips)
    - [ ] Internet connection stable (if cloud deployed)
    - [ ] Backup plan ready

  - [ ] Option B: Recorded demo video
    - [ ] Record 1-2 minute demo
    - [ ] Show upload → processing → results
    - [ ] Embed in presentation

- [ ] Practice presentation:
  - [ ] Time yourself (aim for 8-10 minutes)
  - [ ] Practice transitions
  - [ ] Practice explaining technical concepts clearly
  - [ ] Practice demo (if live)

- [ ] Prepare for Q&A:
  - [ ] Why did you choose LSTM over other architectures?
  - [ ] How does your model handle doubles matches?
  - [ ] What is the biggest limitation of your system?
  - [ ] How could this be deployed in practice?
  - [ ] What would you improve given more time?
  - [ ] How did you ensure the model is not biased?

#### Deliverable
- ✅ Presentation slides (PDF or PPT)
- ✅ Demo ready (live or video)
- ✅ Confident in delivery
- ✅ Prepared for questions

#### Presentation Checklist
- [ ] Slides are visually clear (not too much text)
- [ ] Figures/charts are readable
- [ ] Consistent formatting and color scheme
- [ ] No spelling errors
- [ ] Timing: 8-10 minutes (practiced)
- [ ] Demo tested and working
- [ ] Backup demo video (if doing live demo)
- [ ] Anticipate likely questions
- [ ] Presentation file named: `ITI123_Presentation_YourName.pdf`

#### Presentation Practice Log
```
Practice Run 1 (Date: _____):
- Time: _____ minutes
- Issues: _________
- Improvements needed: _________

Practice Run 2 (Date: _____):
- Time: _____ minutes
- Issues: _________
- Improvements needed: _________

Practice Run 3 (Date: _____):
- Time: _____ minutes
- Ready: ☐ Yes ☐ Need more practice
```

#### Notes
_Add your notes here..._

---

## 🎯 Final Submission
**Date: Feb 26, 2026, before 2359 hrs** ⚠️

### Final Submission Checklist

#### 1. Final Report
- [ ] PDF file prepared
- [ ] File name: `ITI123_Final_YourName.pdf`
- [ ] 15 pages or less (excluding references)
- [ ] All sections complete
- [ ] All figures labeled and referenced
- [ ] All tables formatted
- [ ] All citations included
- [ ] Proofread and polished
- [ ] Submitted to portal before 2359 hrs

#### 2. Code Submission
Choose one:

**Option A: GitHub Repository**
- [ ] Repository is public or accessible to instructors
- [ ] Link included in report
- [ ] README.md is comprehensive
- [ ] All code is well-organized
- [ ] .gitignore excludes large files
- [ ] Repository URL: _________

**Option B: ZIP File**
- [ ] ZIP file size < 5MB
- [ ] Contains all source code
- [ ] Contains README.md
- [ ] Contains requirements.txt
- [ ] Contains model architecture diagram
- [ ] Contains brief file descriptions
- [ ] Excludes videos, data, large models
- [ ] File name: `ITI123_Code_YourName.zip`
- [ ] Submitted to portal before 2359 hrs

#### 3. Code Documentation
Ensure ZIP/repository includes:
- [ ] README.md with:
  - Project overview
  - Installation instructions
  - Usage examples
  - Dataset citation
  - Model architecture
  - Results summary
- [ ] requirements.txt
- [ ] All source code in `src/` folder
- [ ] Sample notebooks in `notebooks/` folder
- [ ] Main app script: `app.py`
- [ ] File structure documentation

#### 4. Presentation Materials (for Feb 25-26)
- [ ] Presentation slides (PDF/PPT)
- [ ] Demo ready (live or video)
- [ ] File name: `ITI123_Presentation_YourName.pdf`

#### 5. Final Verification
- [ ] All files named correctly
- [ ] All deadlines met
- [ ] Submitted to correct portal
- [ ] Received confirmation of submission
- [ ] Backup copies saved

---

## 📊 Risk Management

### Risk 1: Pose Estimation Quality Issues
**Likelihood:** Medium
**Impact:** High

**Mitigation:**
- Test pose extraction on sample clips in Week 2
- Compare MediaPipe vs MoveNet quality
- Implement confidence score filtering
- Validate on diverse clips (lighting, angles)

**Contingency:**
- If pose quality poor (<70% confidence), use coordinate-based approach with ShuttleSet annotations
- Focus on high-quality clips only (filter out poor poses)
- Manual validation and filtering

---

### Risk 2: Model Performance Below Expectations
**Likelihood:** Medium
**Impact:** Medium

**Mitigation:**
- Establish baseline early (Week 3)
- Systematic hyperparameter tuning (Week 5)
- Error analysis to guide improvements
- Multiple architecture experiments

**Contingency:**
- If accuracy <75%, focus report on methodology and insights rather than just performance
- Emphasize learning process and challenges overcome
- Discuss why certain approaches didn't work
- Strong error analysis and future work section

---

### Risk 3: Time Constraints
**Likelihood:** High
**Impact:** Medium

**Mitigation:**
- Start immediately
- Follow weekly milestones strictly
- Prioritize core features
- Use consultation sessions for guidance

**Contingency Plan (Priority Order):**

**Must Have (Essential):**
- Working pose extraction pipeline
- Feature engineering implemented
- LSTM model trained and evaluated
- Basic metrics (accuracy, confusion matrix)
- Milestone Report (Jan 29)
- Final Report (Feb 26)
- Presentation (Feb 25)

**Should Have (Important):**
- Baseline model for comparison
- Hyperparameter tuning
- Error analysis with visualizations
- Gradio deployment (local)
- Comprehensive Final Report

**Nice to Have (Optional):**
- Advanced techniques (attention, multi-task)
- Cloud deployment
- Multiple architecture comparisons
- Extensive ablation studies
- Publication-quality visualizations

**If Time is Running Out:**
- Week 5: Skip advanced techniques (Milestone 5.2), focus on hyperparameter tuning only
- Week 6: Simplify deployment (CLI instead of Gradio)
- Week 6: Streamline Final Report (focus on required sections only)

---

### Risk 4: Computational Resources
**Likelihood:** Medium
**Impact:** Medium

**Mitigation:**
- Use Google Colab Pro (if local GPU unavailable)
- Start with smaller model (fewer LSTM layers)
- Optimize batch size for memory
- Use model checkpointing (save best models)

**Contingency:**
- Train on subset of data (e.g., 15K strokes instead of full dataset)
- Use smaller sequence length (30 frames instead of 75)
- Use GRU instead of LSTM (fewer parameters)
- Train on Colab GPU (free tier: 12 hours/day)

**GPU Options:**
- Local GPU: ☐ Available ☐ Not available
- Google Colab: ☐ Free tier ☐ Pro ☐ Not using
- Cloud GPU: ☐ AWS ☐ GCP ☐ Azure ☐ Not using

---

### Risk 5: Dataset Issues
**Likelihood:** Low
**Impact:** High

**Mitigation:**
- Verify video files and annotations match (Week 1)
- Check for corrupted videos early
- Validate stroke labels with sample clips
- Document any data quality issues

**Contingency:**
- If many videos corrupted: Filter out and proceed with available data
- If annotations have errors: Manual correction for subset, document limitations
- If insufficient Clear/Smash examples: Consider adding one more stroke type (e.g., Drop)

---

## 📈 Progress Tracking

### Weekly Self-Assessment Questions

**Week 1 Check-in (Jan 21):**
- [ ] Can you load videos successfully?
- [ ] Do you have exact counts of Clear/Smash strokes?
- [ ] Can you extract and view sample clips?
- [ ] Are you on schedule?

**Week 2 Check-in (Jan 28):**
- [ ] Can you extract poses from videos reliably?
- [ ] Do pose overlays look correct?
- [ ] Have you created feature vectors?
- [ ] Are you ready for Milestone Report?

**Week 3 Check-in (Feb 4):**
- [ ] Is your baseline model trained?
- [ ] What's the baseline accuracy? _____%
- [ ] Is your LSTM model implemented?
- [ ] Was Milestone Report submitted on time?

**Week 4 Check-in (Feb 11):**
- [ ] Is your LSTM model training successfully?
- [ ] Does it outperform the baseline?
- [ ] Have you identified failure cases?
- [ ] Are you on track for optimization?

**Week 5 Check-in (Feb 18):**
- [ ] Have you tested multiple configurations?
- [ ] What's your best model performance? _____%
- [ ] Is the model ready for deployment?
- [ ] Do you have comprehensive results?

**Week 6 Check-in (Feb 25):**
- [ ] Does your Gradio app work end-to-end?
- [ ] Is your Final Report drafted?
- [ ] Is your presentation ready?
- [ ] Are you confident for presentation day?

---

## 📚 Resources & References

### Essential Tools
- **MediaPipe Pose:** https://google.github.io/mediapipe/solutions/pose
- **MoveNet:** https://www.tensorflow.org/hub/tutorials/movenet
- **TensorFlow/Keras:** https://www.tensorflow.org/
- **PyTorch:** https://pytorch.org/
- **Gradio:** https://gradio.app/
- **MLflow:** https://mlflow.org/
- **OpenCV:** https://opencv.org/

### Dataset
- **ShuttleSet GitHub:** https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet
- **ShuttleSet Paper:** Wang et al., "ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis", KDD 2023
- **ArXiv:** https://arxiv.org/abs/2306.04948

### Key Papers to Cite
1. Wang et al., "ShuttleSet", KDD 2023 (dataset)
2. Lugaresi et al., "MediaPipe: A Framework for Building Perception Pipelines", 2019 (if using MediaPipe)
3. Hochreiter & Schmidhuber, "Long Short-Term Memory", Neural Computation, 1997 (LSTM)
4. Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder", EMNLP 2014 (GRU)

### Tutorials & Guides
- Pose estimation with MediaPipe: https://google.github.io/mediapipe/solutions/pose.html
- LSTM tutorial: https://www.tensorflow.org/guide/keras/rnn
- Gradio quickstart: https://gradio.app/quickstart/

---

## 📞 Support & Consultation

### Course Resources
- **Consultation Sessions:** Use them! Show progress and get feedback
- **Project Mentors:** Available during consultation hours
- **Course Materials:** Review lecture notes on RNNs, sequence modeling

### When to Seek Help
- Stuck on implementation for >4 hours
- Unclear about assignment requirements
- Need guidance on technical decisions
- Want feedback on progress
- Facing unexpected issues

### What to Prepare for Consultation
1. Specific question or issue
2. What you've tried so far
3. Code/error messages (if technical issue)
4. Current progress status

---

## ✅ Success Criteria Reminder

### Minimum Viable Project (Pass)
- ✅ Pose-based features extracted from videos
- ✅ LSTM model trained and evaluated
- ✅ Accuracy > 70%
- ✅ Basic deployment (CLI or simple Gradio)
- ✅ Complete Milestone and Final Reports
- ✅ Presentation delivered

### Good Project (Exceed Expectations)
- ✅ Thorough error analysis
- ✅ Optimized model (Accuracy > 80%)
- ✅ Polished Gradio interface
- ✅ Well-documented code with README
- ✅ Insightful reports with comprehensive results
- ✅ Clear presentation with demo

### Excellent Project (Outstanding)
- ✅ Novel techniques (attention, multi-task learning)
- ✅ High accuracy (> 85%)
- ✅ Cloud deployment
- ✅ Comprehensive analysis and ablation studies
- ✅ Publication-quality reports
- ✅ Exceptional presentation with live demo
- ✅ Valuable insights and contributions

---

## 📝 Personal Notes & Reflections

### Week 1 Reflections
_What went well:_

_Challenges faced:_

_Lessons learned:_

---

### Week 2 Reflections
_What went well:_

_Challenges faced:_

_Lessons learned:_

---

### Week 3 Reflections
_What went well:_

_Challenges faced:_

_Lessons learned:_

---

### Week 4 Reflections
_What went well:_

_Challenges faced:_

_Lessons learned:_

---

### Week 5 Reflections
_What went well:_

_Challenges faced:_

_Lessons learned:_

---

### Week 6 Reflections
_What went well:_

_Challenges faced:_

_Lessons learned:_

---

## 🎓 Final Project Summary

_To be filled after completion:_

**Project Completion Date:** _____

**Final Achievements:**
- Model accuracy: _____%
- Key technical contributions: _________
- Challenges overcome: _________
- Most valuable learning: _________

**What I'm Proud Of:**
1. _________
2. _________
3. _________

**What I Would Do Differently:**
1. _________
2. _________
3. _________

**Future Plans for This Project:**
_________

---

**END OF PROJECT PLAN**

---

## Quick Reference: Key Commands

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run data processing
python src/data_processing/extract_clips.py
python src/data_processing/extract_poses.py
python src/data_processing/feature_engineering.py

# Train models
python src/models/train.py --model baseline
python src/models/train.py --model lstm --epochs 50

# Launch Gradio app
python app.py

# Start Jupyter
jupyter notebook
```

## Quick Reference: File Paths

```
Data:
- Raw videos: data/raw_videos/
- Annotations: data/annotations/
- Processed clips: data/processed/clips/
- Poses: data/processed/poses/
- Features: data/processed/features/

Models:
- Saved models: models/
- Best model: models/final_model.h5

Results:
- Plots: outputs/plots/
- Reports: outputs/reports/
```
