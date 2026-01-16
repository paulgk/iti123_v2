# AI Badminton Coach - Quick Start

Get started in 3 simple steps.

---

## 1. Install

```bash
pip install -r requirements.txt
python diagnose.py
```

Expected output: `✅ ALL CHECKS PASSED`

---

## 2. Analyze Video

```bash
python analyze_video.py your_video.mp4 Clear
# or
python analyze_video.py your_video.mp4 Smash
```

Results saved to: `outputs/video_analysis/<video_name>/`

---

## 3. Web Interface (Recommended)

```bash
streamlit run src/deployment/streamlit_app.py
```

Then:
1. Select "Upload Video" mode
2. Upload your .mp4 file
3. Choose stroke type (Clear/Smash)
4. Click "Process Video"
5. Review results in 3 tabs

---

## Troubleshooting

**Mutex Error?**
```bash
pip install protobuf==3.20.3
python diagnose.py
```

**More Help?**
See [README.md](README.md) for full documentation.

---

Last updated: January 16, 2026
