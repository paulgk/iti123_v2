# AI Badminton Coaching App - v1.0 Milestone

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

### Active

- [ ] Validate system on external videos (beyond ShuttleSet dataset)
- [ ] Test feedback accuracy (biomechanically correct suggestions)
- [ ] Test severity ranking (critical issues prioritized correctly)
- [ ] Finalize LaTeX milestone report (align with rubrics)
- [ ] Document benchmark methodology in report
- [ ] Address rubric criteria systematically
- [ ] Prepare submission package (report + code + demo)

### Out of Scope

- ML classification improvements (acknowledged limitation for v1.0)
- Backhand stroke benchmarks (forehand-only for v1.0)
- Additional stroke types beyond Clear & Smash (future work)
- Mobile app or deployment infrastructure (v2.0+)
- Real-time video analysis (v2.0+)

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
- **Dataset**: Limited to ShuttleSet (cannot gather new professional data)
- **Stroke Types**: Clear & Smash only (no time for additional stroke types)
- **Timeline**: Milestone due next week or later (but want to submit confidently)
- **Validation**: Need ground truth data to validate feedback accuracy
- **Academic**: Report must follow LaTeX format and address rubrics

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Pivot from ML to benchmarks | Classification accuracy too low, benchmark approach more interpretable and actionable | ✓ Good - System provides useful feedback |
| Forehand-only benchmarks | Limited time, filtered backhand from dataset to improve benchmark quality | ⚠️ Revisit - Limits applicability to half of strokes |
| Streamlit for interface | Rapid prototyping, easy video upload, good for demo | ✓ Good - Works well for milestone |
| ShuttleSet dataset | Only available professional badminton stroke dataset with annotations | ✓ Good - High quality data |
| No automated testing | Time pressure, manual validation sufficient for v1.0 | ⚠️ Revisit - Risky for future iterations |

---
*Last updated: 2026-01-21 after initialization*
