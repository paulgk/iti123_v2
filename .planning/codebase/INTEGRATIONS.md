# External Integrations

**Last Updated**: 2026-01-21
**Codebase**: AI Badminton Coaching System v2.0

---

## Overview

This system has minimal external dependencies. It operates primarily as a standalone application with local data processing and optional LLM enhancement.

---

## External APIs

### OpenAI API (Optional)

**Status**: Optional feature, not required for core functionality
**Location**: [`src/coaching/llm_enhancer.py`](src/coaching/llm_enhancer.py)

**Purpose**:
- Enhances rule-based coaching feedback with natural language generation
- Provides conversational, personalized feedback
- Generates creative practice drills

**Model Used**: `gpt-4o-mini` (fast, cost-effective)

**Authentication**:
- Environment variable: `OPENAI_API_KEY`
- Or passed as parameter to `LLMCoachingEnhancer(api_key=...)`
- Not required if LLM enhancement is disabled

**API Calls**:
```python
# New OpenAI API (v1.0+)
client = OpenAI(api_key=api_key)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    temperature=0.7
)

# Fallback for legacy API
openai.api_key = api_key
response = openai.ChatCompletion.create(...)
```

**Error Handling**:
- Falls back to template-based feedback if LLM unavailable
- Retry logic for transient failures
- Graceful degradation if `openai` package not installed

**Current Usage**: Experimental feature, not used in main workflows ([`analyze_video.py`](analyze_video.py), Streamlit app)

---

## Databases

**Status**: None

This system uses **file-based storage only**:
- Pose sequences: Pickle files in `data/processed/poses/`
- Features: Pickle files in `data/processed/features/`
- Metadata: CSV file at `data/processed/clips/clips_metadata.csv`
- Annotations: CSV files in `data/annotations/`

**No database engine** (SQL, NoSQL, etc.) is used.

---

## Authentication & Authorization

**Status**: None

The system has no authentication:
- Streamlit app: Open access on localhost
- Gradio app: Open access on localhost
- No user accounts, sessions, or access control

**Security Note**: Designed for local/research use only. Would need auth layer for production deployment.

---

## Cloud Services

**Status**: None (except optional OpenAI)

No cloud integrations:
- No cloud storage (AWS S3, GCS, Azure Blob)
- No cloud compute (Lambda, Cloud Functions)
- No cloud ML services (AWS SageMaker, Vertex AI)

**Deployment**: Local only (or manual cloud VM deployment)

---

## Data Sources

### ShuttleSet Dataset

**Type**: Static dataset (not API/service)
**Source**: [arXiv:2306.04948](https://arxiv.org/abs/2306.04948)
**Access Method**: Manual download and placement in `data/` directory
**Format**: Video clips (.mp4) + CSV annotations

**Citation**:
```bibtex
@article{wang2023shuttleset,
  title={ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis},
  author={Wang, Wei-Yao and Huang, Yu-Chuan and Ik, Tsi-Ui and Peng, Wen-Chih},
  journal={arXiv preprint arXiv:2306.04948},
  year={2023}
}
```

**Dataset Stats**:
- 4,983 clips processed
- 3,347 strokes used for benchmarks (Clear + Smash, forehand only)
- Professional singles matches

**Integration**: Offline, pre-processed into features

---

## Third-Party Libraries (Not APIs)

### MediaPipe Pose

**Type**: Client-side library (not web service)
**Provider**: Google
**Documentation**: https://google.github.io/mediapipe/solutions/pose.html

**Purpose**: Pose estimation from video frames
**Model**: Pre-trained on-device model (no external API calls)
**Output**: 33 keypoints per detected person

**Not a web service** - runs locally, no network calls.

---

## Webhooks & Callbacks

**Status**: None

No webhook endpoints or callback mechanisms.

---

## Message Queues & Event Buses

**Status**: None

No asynchronous messaging:
- No RabbitMQ, Kafka, Redis Pub/Sub, etc.
- All processing is synchronous

---

## External Storage

**Status**: None

All data stored locally:
- Video clips: `data/processed/clips/`
- Pose data: `data/processed/poses/`
- Features: `data/processed/features/`
- Outputs: `outputs/video_analysis/`

**Google Colab Integration** (Indirect):
- Some debug scripts ([`debug_splits_colab.py`](debug_splits_colab.py)) reference Colab paths
- Used for accessing large dataset files during development
- Not a production integration

---

## Monitoring & Analytics

**Status**: None

No external monitoring:
- No Sentry, Datadog, New Relic, etc.
- No analytics tracking (Google Analytics, Mixpanel)
- No error reporting services

**Experiment Tracking**:
- MLflow configured but not actively used in v2.0
- Would track experiments if model training workflow activated

---

## CI/CD & Deployment

**Status**: None

No automated pipelines:
- No GitHub Actions, CircleCI, Jenkins
- No automated testing on commit
- No automated deployment

**Deployment Method**: Manual
1. Clone repo
2. `pip install -r requirements.txt`
3. `pip install protobuf==3.20.3` (critical fix)
4. Run `streamlit run src/deployment/streamlit_app.py`

---

## Potential Future Integrations

Based on codebase structure and commented code:

1. **PyTorch** (Alternative to TensorFlow)
   - Commented in `requirements.txt`
   - Would require refactoring model code

2. **Testing Services**
   - pytest, pytest-cov (commented in requirements)
   - Would enable CI/CD integration

3. **Code Quality Services**
   - Black, Flake8, MyPy (commented in requirements)
   - Could integrate with GitHub Actions

4. **Video Storage**
   - Cloud storage for uploaded videos (AWS S3, GCS)
   - Would enable web-scale deployment

5. **Database**
   - PostgreSQL or MongoDB for user profiles, analysis history
   - Required for multi-user production deployment

6. **Authentication Provider**
   - Auth0, Firebase Auth, Clerk
   - Required for production deployment

---

## Network Requirements

**Development**: Minimal
- Internet required only for: `pip install` (one-time)
- Optional: OpenAI API calls (if LLM enhancement enabled)

**Runtime**: None
- System operates fully offline after installation
- No external API dependencies for core features

**Deployment**: Localhost only
- Streamlit: `http://localhost:8501`
- Gradio: `http://localhost:7860`
- No production hosting configured

---

## Security Considerations

1. **No input sanitization** - assumes trusted video files
2. **No rate limiting** - vulnerable to abuse if exposed publicly
3. **No HTTPS** - local HTTP only
4. **No API key encryption** - OpenAI key stored in plaintext env var
5. **No access logs** - no audit trail of analysis requests

**Recommendation**: Do NOT expose to public internet without security hardening.

---

## Integration Testing

**Status**: Not present

No integration tests for:
- OpenAI API calls
- File I/O operations
- Video processing pipeline

**Testing Method**: Manual verification via [`diagnose.py`](diagnose.py)
