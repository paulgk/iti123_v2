# Badminton Shot Coach

An AI system that watches a badminton video and classifies the shot type. Built as an ITI123 Generative AI & Deep Learning project.

**Live demo:** [huggingface.co/spaces/paulgk/badminton-shot](https://huggingface.co/spaces/paulgk/badminton-shot)

---

## What It Does

Upload a short video clip of a badminton shot and the app returns:

- Predicted shot type (Clear, Drive, Drop, Lift, or Smash)
- Confidence score and probability breakdown for all 5 classes
- Video metadata (duration, frame count, which frames were sampled)
- Automatic warnings (video too long, low confidence, etc.)

---

## How It Works

The model is a **ResNet18 + BiLSTM** pipeline:

1. **ResNet18** extracts visual features from 16 frames sampled evenly across the clip
2. **BiLSTM** tracks how those features change over time to capture motion patterns
3. Trained on 22,302 clips from the [ShuttleSet](https://arxiv.org/abs/2306.04948) professional match dataset
4. Achieves **74.6% accuracy** across 5 shot types on a held-out test set

An earlier approach using MediaPipe body pose tracking (skeleton keypoints) was tried first but achieved only ~50% accuracy — no better than random — because players use nearly identical body positions for Clear and Smash shots. The racket angle is the key discriminator, and body pose data cannot capture it.

## Project Structure

```
iti123_v2/
├── notebooks/                # Training notebooks (Colab)
│   ├── badminton_video_training_colab_v3_optimized.ipynb  # Final training run
│   └── ...
│
├── src/
│   └── data_processing/      # Pose extraction (Phase 1 / archival)
│
├── outputs/
│   ├── results_optionA/      # Trained model weights (best_model.pth)
│   └── reports/              # Final project reports (.tex / .pdf)
│
└── data/
    └── processed/            # Extracted features and clip metadata
```

---

## Model Performance

| Shot Type | Precision | Recall | F1    | Test Clips |
|-----------|-----------|--------|-------|------------|
| Clear     | 67.6%     | 89.2%  | 76.9% | 535        |
| Drive     | 58.3%     | 61.6%  | 59.9% | 740        |
| Drop      | 90.7%     | 74.6%  | 81.9% | 1,518      |
| Lift      | 71.0%     | 75.7%  | 73.3% | 966        |
| Smash     | 76.5%     | 75.8%  | 76.1% | 724        |
| **Overall** | **72.8%** | **75.4%** | **73.6%** | **4,483** |

Drive is the hardest class — it's visually similar to Lift and also has the least training data.

---

## Training

Training was done on Google Colab (T4 GPU). The final notebook is [`notebooks/badminton_video_training_colab_v3_optimized.ipynb`](notebooks/badminton_video_training_colab_v3_optimized.ipynb).

Key training decisions:
- **Focal Loss** to handle the 9.2:1 class imbalance (Drop >> Clear)
- **Class weights** giving minority classes proportionally more importance
- **Two-stage training**: freeze ResNet18 backbone first, then unfreeze and fine-tune end-to-end
- **16 frames** sampled evenly per clip at 224×224 resolution
- **Early stopping** with patience=10 on validation accuracy

---

## Dataset

**ShuttleSet** — Wei-Yao Wang et al. (2023)

```bibtex
@article{wang2023shuttleset,
  title={ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis},
  author={Wang, Wei-Yao and Huang, Yu-Chuan and Ik, Tsi-Ui and Peng, Wen-Chih},
  journal={arXiv preprint arXiv:2306.04948},
  year={2023}
}
```

22,302 labeled clips from 40 professional singles matches, filmed from broadcast side-court camera angles.

---

## Acknowledgments

Code development and report writing assisted by Claude Code (Anthropic). All model training, evaluation, and application development by Paul George Karippaparambil.

---

**Last updated:** February 2026
