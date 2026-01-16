# Save this as baseline_model_fixed.py in Colab
import sys
sys.path.insert(0, '/content/drive/MyDrive/ITI123')

# Force the correct paths
import pickle
from pathlib import Path

SPLITS_DIR = Path("/content/drive/MyDrive/ITI123/data/processed/splits")

print("FORCED PATHS - Loading from:", SPLITS_DIR)

with open(SPLITS_DIR / "train_data.pkl", 'rb') as f:
    train = pickle.load(f)

print(f"Train samples: {len(train['y'])}")
print(f"X shape: {train['X'].shape}")
print(f"X_stat_raw shape: {train['X_stat_raw'].shape}")

# If this prints 3554, then we know the files are correct
# and the issue is with baseline_model.py's path resolution
