# AI-Based Badminton Stroke Technique Assessment

**ITI123 Generative AI & Deep Learning Project**

An AI system that analyzes badminton overhead stroke execution (Clear and Smash) using pose-based deep learning from video footage.

---

## 🎯 Project Overview

This project develops a deep learning system that:
- Extracts human pose keypoints from badminton match videos
- Engineers biomechanical features from pose sequences
- Classifies overhead strokes (Clear vs Smash) using LSTM/GRU models
- Provides objective technique assessment feedback
- Deploys as an interactive web application

**Dataset:** ShuttleSet - 36,492 stroke-level annotations from 44 professional matches

**Approach:** Pose Estimation → Feature Engineering → Temporal Deep Learning → Classification

---

## 📁 Project Structure

```
iti123_v2/
├── data/
│   ├── raw_videos/           # Original ShuttleSet videos
│   ├── annotations/          # CSV annotation files
│   └── processed/            # Processed data (clips, poses, features)
├── notebooks/                # Jupyter notebooks for exploration
├── src/                      # Source code
│   ├── data_processing/      # Data preprocessing scripts
│   ├── models/               # Model architectures and training
│   ├── evaluation/           # Evaluation metrics and visualization
│   └── deployment/           # Deployment code
├── experiments/              # MLflow experiment tracking
├── models/                   # Saved model checkpoints
├── outputs/                  # Results, plots, reports
├── app.py                    # Gradio web application
├── requirements.txt          # Python dependencies
└── PROJECT_PLAN.md          # Detailed implementation plan
```

---

## 🚀 Quick Start

### 1. Setup Environment

**On macOS/Linux:**
```bash
# Make setup script executable
chmod +x setup_project.sh

# Run setup script
./setup_project.sh

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**On Windows:**
```cmd
# Run setup script
setup_project.bat

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

1. Place raw video files in `data/raw_videos/`
2. Download ShuttleSet annotations and place in `data/annotations/`
3. Run data exploration notebook: `notebooks/01_data_exploration.ipynb`

### 3. Run Pipeline

```bash
# Extract clips from videos
python src/data_processing/extract_clips.py

# Extract pose keypoints
python src/data_processing/extract_poses.py

# Engineer features
python src/data_processing/feature_engineering.py

# Train baseline model
python src/models/train.py --model baseline

# Train LSTM model
python src/models/train.py --model lstm --epochs 50
```

### 4. Launch Web App

```bash
python app.py
```

---

## 📊 Current Status

**Phase:** [Update as you progress]
- [x] Project setup complete
- [ ] Data preprocessing
- [ ] Baseline model
- [ ] Deep learning model
- [ ] Deployment
- [ ] Final report

**Latest Results:**
- Baseline Accuracy: ___%
- LSTM Accuracy: ___%

*(Update this section as you make progress)*

---

## 📚 Dataset Citation

```bibtex
@inproceedings{wang2023shuttleset,
  title={ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis},
  author={Wang, Wei-Yao and Huang, Yung-Chang and Ik, Tsi-Ui and Peng, Wen-Chih},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2023}
}
```

**Dataset:** [ShuttleSet on GitHub](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet)

**License:** MIT

---

## 🛠️ Technologies Used

- **Pose Estimation:** MediaPipe Pose / MoveNet
- **Deep Learning:** TensorFlow/Keras (LSTM/GRU)
- **Computer Vision:** OpenCV
- **Experiment Tracking:** MLflow
- **Deployment:** Gradio
- **Data Processing:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib, Seaborn

---

## 📝 Documentation

- **Detailed Plan:** See [PROJECT_PLAN.md](PROJECT_PLAN.md)
- **Milestone Report:** [Link to be added after Jan 29]
- **Final Report:** [Link to be added after Feb 26]
- **Presentation:** [Link to be added after Feb 25]

---

## 📅 Key Milestones

| Milestone | Due Date | Status |
|-----------|----------|--------|
| Milestone Report | Jan 29, 2026 | ⏳ Pending |
| Final Report | Feb 26, 2026 | ⏳ Pending |
| Presentation | Feb 25-26, 2026 | ⏳ Pending |

---

## 🎓 Course Information

**Course:** ITI123 Generative AI & Deep Learning Project (2025S2)

**Institution:** Nanyang Polytechnic

**Focus Area:** Model Development (Training, Fine-tuning)

---

## 📧 Contact

**Student:** Paul

**Project Type:** Individual Project

---

## 📄 License

This project uses the MIT-licensed ShuttleSet dataset. Project code is for academic purposes.

---

## 🙏 Acknowledgments

- ShuttleSet dataset authors (Wang et al., KDD 2023)
- Advanced Database System Laboratory, National Yang Ming Chiao Tung University
- Course instructors and mentors

---

**Last Updated:** [Add date as you update]

**Next Action:** [Update with your next immediate step]
