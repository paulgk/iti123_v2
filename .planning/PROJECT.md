# AI Badminton Coaching App

## What This Is

An AI-powered badminton coaching system that analyzes user-submitted stroke videos (Clear & Smash) using MediaPipe pose estimation and provides personalized technique feedback based on professional benchmarks derived from 3,347 professional strokes.

## Core Value

Analyze an input video and give accurate, actionable feedback on how to improve stroke technique.

## Requirements

### Validated

- ✓ MediaPipe pose extraction (33 keypoints) — existing
- ✓ Feature engineering (427 biomechanical features) — existing
- ✓ Professional benchmark ranges (Clear & Smash, forehand) — existing
- ✓ Coaching feedback generation (severity classification, drills) — existing
- ✓ Streamlit web interface (video upload → analysis → feedback) — existing
- ✓ Visualization system (radar charts, bar charts, score gauges) — existing

## Current Milestone: v1.1 Coach-Informed ML + Colab Infrastructure

**Goal:** Improve ML classification accuracy through coach-informed feature engineering and establish Colab Enterprise workflow for scalable training.

**Target features:**
- Colab Enterprise infrastructure with Git LFS for videos
- Coach-informed feature expansion (beyond 427 features)
- Retrained ML classification model with improved accuracy
- Extended stroke type coverage (Clear, Smash, Drop, Drive, Net shots)
- Bidirectional git ↔ Colab sync workflow

### Active

- [ ] Set up Git LFS for video storage
- [ ] Configure Colab Enterprise environment for script execution
- [ ] Create git ↔ Colab sync workflow (data pull, output push)
- [ ] Research coaching biomechanics from literature/videos
- [ ] Identify and implement coach-suggested features
- [ ] Expand feature set beyond current 427 features
- [ ] Retrain classification model with improved features
- [ ] Expand stroke type support (Drop, Drive, Net shots)
- [ ] Validate classification accuracy improvements
- [ ] Assess feedback quality improvements

### Out of Scope

- Jupyter notebooks in Colab (using terminal scripts only)
- Direct coach consultation (using literature/videos instead)
- Backhand-specific benchmarks (v1.1 still forehand-focused for new strokes)
- Mobile app or deployment infrastructure (v2.0+)
- Real-time video analysis (v2.0+)
- Automated testing framework (future work)

## Context

**Technical Environment:**
- Python 3.10, TensorFlow 2.15.x (for MediaPipe compatibility)
- MediaPipe 0.10.9 for pose estimation (requires protobuf 3.20.3)
- Streamlit for web interface
- Dataset: ShuttleSet (4,983 clips, 3,347 Clear + Smash forehand strokes)

**Current System State:**
- Codebase is functional and documented (see `.planning/codebase/`)
- Benchmark-based analysis (no ML classification in production)
- Tested on dataset clips, not yet validated on external videos
- LaTeX report drafted in `outputs/reports/`, needs refinement

**Milestone Context:**
- Academic submission with rubrics (PDF in `outputs/reports/`)
- Deadline: Next week or later (reasonable time for validation + polish)
- Primary focus: Demonstrate benchmark methodology is sound
- Secondary focus: Show system works end-to-end (video → feedback)

**Known Issues:**
- ML classification accuracy too low (not used in production)
- Benchmarks are forehand-only (backhand strokes analyzed incorrectly)
- No systematic validation on external videos yet
- Large file removed from git history (105MB train_data.pkl) - requires force push

## Constraints

- **Tech Stack**: Must use existing Python/MediaPipe/Streamlit stack
- **Colab Environment**: Colab Enterprise, terminal-based script execution only
- **Dataset**: Limited to ShuttleSet (cannot gather new professional data)
- **Coach Input**: Indirect via literature/videos (no direct consultation)
- **Git Storage**: Must use Git LFS for video files to manage repo size
- **Validation**: Classification accuracy and feedback quality metrics

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Pivot from ML to benchmarks | Classification accuracy too low, benchmark approach more interpretable and actionable | ✓ Good - System provides useful feedback |
| Forehand-only benchmarks | Limited time, filtered backhand from dataset to improve benchmark quality | ⚠️ Revisit - Limits applicability to half of strokes |
| Streamlit for interface | Rapid prototyping, easy video upload, good for demo | ✓ Good - Works well for milestone |
| ShuttleSet dataset | Only available professional badminton stroke dataset with annotations | ✓ Good - High quality data |
| No automated testing | Time pressure, manual validation sufficient for v1.0 | ⚠️ Revisit - Risky for future iterations |

---
*Last updated: 2026-01-29 after v1.1 milestone start*
