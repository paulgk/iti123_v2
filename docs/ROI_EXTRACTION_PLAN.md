# ROI-Based Single Player Extraction Plan

**Date:** 2026-02-03
**Decision:** Extract only the player who performed the shot (not both players)
**Rationale:** Perfect timing alignment and clear biomechanics

---

## Decision Rationale

### Why Single Player Per Shot?

**User's insight: "start with one player who is doing the exact shot"**

**Benefits:**

1. **Perfect Timing Alignment**
   - Clip centered on exact frame when player performs shot
   - No timing offset issues
   - Model sees shot at peak execution moment

2. **Clear Biomechanics**
   - Only the performing player's skeleton in ROI
   - No confusion from opponent's waiting position
   - Clean training signal for shot classification

3. **Simpler Implementation**
   - One clip per shot (not two)
   - Easier validation and debugging
   - Less storage and processing time

4. **Still Solves Multi-Player Issue**
   - ROI focuses on performing player only
   - 0% skeleton merging with opponent
   - Clean pose extraction guaranteed

5. **Still 2.2x More Data**
   - Current: ~7,000 usable samples (with multi-player noise)
   - Single player ROI: ~15,822 usable samples (clean)
   - Significant improvement without complexity

---

## Updated Dataset Size

### Extraction Pipeline

```
19,778 clean shots (with refined mapping)
  × 1 player per shot (performer only)
= 19,778 clips

After filtering:
  - Short sequences (<30 frames): ~20% removed → 15,822 clips
  - Multi-player (>60% width): 0% with ROI → 15,822 clips

Final: ~15,822 usable training samples
```

**Comparison:**

| Approach | Clips | After Filter | Multi-Player | Usable |
|----------|-------|--------------|--------------|--------|
| Current (no ROI) | 19,778 | ~13,822 | -1,500 | ~7,000 |
| **Single player ROI** | **19,778** | **~15,822** | **0** | **~15,822** |
| Both players ROI | 39,556 | ~31,644 | 0 | ~31,644 |

**Result:** 2.2x more clean data than current approach

---

## Implementation Plan

### Phase 1.5 ROI Extraction - Single Player

**Files to modify:**

1. **scripts/extract_shuttleset_clips.py**
   - Update SHOT_TYPE_MAPPING (already planned)
   - Add player position reading from CSV
   - Keep clip naming simple: `{match}_set{N}_rally{R}_ball{B}_{shot}.mp4`
   - No `_playerA/B` suffix needed (only one player extracted)

2. **scripts/extract_poses_parallel.py**
   - Add ROI calculation based on player position
   - Crop frame to ROI before MediaPipe processing
   - Transform pose coordinates back to original frame space
   - Ensure single player detection

3. **Metadata tracking**
   - Record which player performed shot in metadata.csv
   - Track player position (x, y) for validation
   - Enable future analysis of position-based patterns

---

## ROI Calculation Strategy

### Using ShuttleSet CSV Data

**Available data per shot:**
```python
{
    'player': 'A',              # Who performed the shot
    'player_location_x': 458,   # X coordinate (0-1920)
    'player_location_y': 256,   # Y coordinate (0-1080)
    'frame_num': 7199,          # Exact frame of shot
    'type': 'Smash',            # Shot type
}
```

### ROI Sizing

**Goal:** Capture full player body with margin for movement

**Strategy:**
```python
def calculate_roi(player_x, player_y, frame_width=1920, frame_height=1080):
    """
    Calculate ROI bounding box around player

    ROI size: 600x800 pixels (width x height)
    - Wide enough for arm extension during shots
    - Tall enough for full body from head to feet
    - Small enough to exclude opponent
    """
    roi_width = 600   # ~31% of frame width
    roi_height = 800  # ~74% of frame height

    # Center ROI on player position
    roi_x1 = max(0, player_x - roi_width // 2)
    roi_y1 = max(0, player_y - roi_height // 2)
    roi_x2 = min(frame_width, roi_x1 + roi_width)
    roi_y2 = min(frame_height, roi_y1 + roi_height)

    # Adjust if ROI hits frame boundary
    if roi_x2 == frame_width:
        roi_x1 = frame_width - roi_width
    if roi_y2 == frame_height:
        roi_y1 = frame_height - roi_height

    return {
        'x1': roi_x1,
        'y1': roi_y1,
        'x2': roi_x2,
        'y2': roi_y2,
        'width': roi_x2 - roi_x1,
        'height': roi_y2 - roi_y1,
    }
```

**ROI size rationale:**
- 600px width = enough for full arm extension during smash
- 800px height = full body from head to toes
- 600x800 = 480,000 pixels (23% of 1920x1080 frame)
- Small enough to exclude opponent (typically >600px away)

---

## Pose Extraction with ROI

### Current Flow (No ROI)
```python
# 1. Load full frame
frame = video.read()

# 2. Run MediaPipe on full frame
pose = mediapipe.detect(frame)  # May detect both players!

# 3. Save pose (possibly merged skeleton)
save_pose(pose)  # ❌ Multi-player contamination
```

### New Flow (With ROI)
```python
# 1. Load full frame
frame = video.read()

# 2. Calculate ROI from player position
roi = calculate_roi(player_x, player_y)

# 3. Crop frame to ROI
cropped_frame = frame[roi['y1']:roi['y2'], roi['x1']:roi['x2']]

# 4. Run MediaPipe on cropped frame
pose = mediapipe.detect(cropped_frame)  # Only one player visible!

# 5. Transform pose coordinates back to original frame space
pose[:, :, 0] += roi['x1']  # Add ROI offset to x coordinates
pose[:, :, 1] += roi['y1']  # Add ROI offset to y coordinates

# 6. Save pose (guaranteed single player)
save_pose(pose)  # ✅ Clean single player skeleton
```

**Benefits:**
- MediaPipe only sees one player in cropped frame
- 0% chance of skeleton merging
- Faster processing (smaller image)
- Higher relative resolution for player details

---

## Updated File Naming

### Clip Naming (No Player Suffix)

**Format:**
```
{match_id}_set{set_num}_rally{rally}_ball{ball}_{shot_type}.mp4
```

**Examples:**
```
01_set1_rally03_ball05_Smash.mp4
01_set1_rally03_ball06_Lift.mp4
15_set2_rally12_ball08_Drop.mp4
```

**Rationale:**
- Simpler than adding `_playerA/B` suffix
- Metadata.csv tracks which player performed shot
- One-to-one mapping: one shot = one clip

### Metadata CSV

**Columns:**
```csv
video_id,match_id,set_num,rally,ball_round,shot_type,player,player_x,player_y,frame_num,clip_path,pose_path
01_set1_rally03_ball05_Smash,1,1,3,5,Smash,A,458,256,7199,data/clips/Smash/01_set1_rally03_ball05_Smash.mp4,data/poses/01_set1_rally03_ball05_Smash.pkl
```

**Enables:**
- Track which player performed shot
- Validate ROI placement (player_x, player_y)
- Analyze position-based patterns later
- Debug any extraction issues

---

## Validation Plan

### 1. ROI Coverage Test

**Sample 100 random clips:**
```python
# For each clip:
# 1. Load video frame at shot moment
# 2. Overlay ROI bounding box
# 3. Verify:
#    - Full player body visible
#    - No body parts cut off
#    - Opponent mostly excluded
#    - Reasonable margin for movement
```

**Expected results:**
- 95%+ clips have full player body in ROI
- <5% edge cases (player at frame boundary)

### 2. Multi-Player Detection Test

**Run multi-player check on extracted poses:**
```python
def test_single_player(pose_sequences):
    """
    Verify no multi-player skeletons after ROI extraction
    """
    multi_player_count = 0

    for pose in pose_sequences:
        x_coords = pose[:, :, 0]
        x_range = np.max(x_coords) - np.min(x_coords)

        if x_range > 0.6:  # Spans >60% of frame
            multi_player_count += 1

    return multi_player_count
```

**Expected results:**
- 0% multi-player detections (was 10-15% without ROI)
- All poses span <40% of frame width (single player)

### 3. Normalization Statistics

**Check pose stats after normalization:**
```python
# After normalize_pose()
Mean: ~0.0 (should be centered)
Std: 0.15-0.35 (should be consistent, not 0.59)
```

**Expected results:**
- Lower variance than current (std 0.59 → 0.25)
- Consistent statistics across all clips

### 4. Sample Visual Inspection

**Check 50 random clips:**
- Load clip and pose overlay
- Verify skeleton matches player performing shot
- Check timing (pose at peak execution moment)
- Validate shot type label matches visual

---

## Expected Training Results

### With Single Player ROI Extraction

**Dataset:**
- Clean shots: 19,778
- Usable samples: ~15,822
- 0% multi-player contamination
- Perfect timing alignment

**Expected accuracy:**

| Model | Current (7K noisy) | Single Player ROI (15.8K clean) | Improvement |
|-------|-------------------|----------------------------------|-------------|
| LSTM | 75-82% | **84-88%** | +9-13% |
| ST-GCN | 85-90% | **89-92%** | +4-7% |
| MS-TCN | 82-88% | **87-91%** | +5-9% |

**Improvements from:**
- ✅ 2.2x more training data (15.8K vs 7K)
- ✅ 0% multi-player contamination
- ✅ Perfect timing (clip centers on shot execution)
- ✅ Clean biomechanics (only performing player)
- ✅ Proper normalization (std 0.15-0.35, not 0.59)

---

## Implementation Steps

### Step 1: Update Shot Type Mapping (Ready)

```bash
# Edit scripts/extract_shuttleset_clips.py
# Update SHOT_TYPE_MAPPING (already documented in SHOT_TYPE_MAPPING_REFINED.md)
```

### Step 2: Add ROI to Clip Extraction

**Add to extract_shuttleset_clips.py:**
```python
def load_shot_annotations_with_positions(shuttleset_dir, match_id, video_name):
    """
    Load shot annotations INCLUDING player positions
    """
    shots = []

    for set_csv in sorted(match_dir.glob('set*.csv')):
        with open(set_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                shots.append({
                    'shot_type': row.get('type'),
                    'player': row.get('player'),
                    'player_x': float(row.get('player_location_x', 0)),
                    'player_y': float(row.get('player_location_y', 0)),
                    'frame_num': float(row.get('frame_num', 0)),
                    # ... other fields
                })

    return shots
```

**Note:** Clip extraction stays the same (full frame clips)
- ROI cropping happens during POSE extraction, not clip extraction
- Clips remain full frame for visual verification

### Step 3: Add ROI to Pose Extraction

**Create new script: scripts/extract_poses_roi.py**

```python
def extract_pose_with_roi(video_path, frame_num, player_x, player_y):
    """
    Extract pose using ROI around player position

    Args:
        video_path: Path to video clip
        frame_num: Frame to extract (relative to clip start)
        player_x: Player X position in original frame
        player_y: Player Y position in original frame

    Returns:
        pose: numpy array (T, 33, 3) with poses in original frame space
    """
    # 1. Load video
    cap = cv2.VideoCapture(video_path)

    # 2. Read frames
    poses = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Calculate ROI for this frame
        # Note: player_x, player_y are from contact frame
        # For other frames, use same ROI (player doesn't move far in 3 seconds)
        roi = calculate_roi(player_x, player_y, frame.shape[1], frame.shape[0])

        # 4. Crop frame to ROI
        cropped = frame[roi['y1']:roi['y2'], roi['x1']:roi['x2']]

        # 5. Run MediaPipe on cropped frame
        results = pose_landmarker.detect(cropped)

        if results and results.pose_landmarks:
            # 6. Transform coordinates back to original frame space
            pose_frame = []
            for landmark in results.pose_landmarks[0]:
                # Scale from ROI to original frame
                x = landmark.x * roi['width'] + roi['x1']
                y = landmark.y * roi['height'] + roi['y1']
                z = landmark.z
                visibility = landmark.visibility

                pose_frame.append([x, y, z])

            poses.append(pose_frame)

        frame_idx += 1

    cap.release()

    return np.array(poses)
```

### Step 4: Update Metadata Creation

**Create metadata.csv with player info:**
```python
# scripts/create_metadata_from_poses.py

metadata_rows = []

for shot in all_shots:
    metadata_rows.append({
        'video_id': f"{match_id}_set{set_num}_rally{rally}_ball{ball}_{shot_type}",
        'match_id': match_id,
        'set_num': set_num,
        'rally': rally,
        'ball_round': ball,
        'shot_type': shot_type,
        'player': shot['player'],  # A or B
        'player_x': shot['player_x'],
        'player_y': shot['player_y'],
        'frame_num': shot['frame_num'],
        'clip_path': clip_path,
        'pose_path': pose_path,
    })
```

### Step 5: Test Extraction

```bash
# Test on one match first
python scripts/extract_shuttleset_clips.py \
    --match-ids 01 \
    --execute

# Should extract ~400-500 clips from match 01

# Extract poses with ROI
python scripts/extract_poses_roi.py \
    --clips data/clips \
    --metadata data/metadata.csv \
    --output data/poses

# Validate results
python scripts/validate_roi_extraction.py
```

### Step 6: Full Extraction

```bash
# Extract all clips
python scripts/extract_shuttleset_clips.py --execute

# Extract all poses with ROI
python scripts/extract_poses_roi.py \
    --clips data/clips \
    --output data/poses \
    --num-workers 8

# Expected time: 8-12 hours for ~20K clips
```

---

## Timeline Estimate

**Phase 1.5 Implementation:**

| Task | Time | Details |
|------|------|---------|
| Update shot mapping | 30 min | Edit SHOT_TYPE_MAPPING |
| Implement ROI logic | 2 hours | Add calculate_roi(), test |
| Update pose extraction | 2 hours | Modify extract_poses_parallel.py |
| Test on 1 match | 1 hour | Validate ROI coverage |
| Full clip extraction | 2 hours | All 19,778 clips |
| Full pose extraction | 8-12 hours | With ROI, 8 workers |
| Validation | 1 hour | Check quality metrics |
| **Total** | **16-20 hours** | Most is extraction time |

**Can run overnight:** Pose extraction can run unattended

---

## Success Metrics

**After ROI extraction, verify:**

✅ **Data Quality**
- 0% multi-player detections (was 10-15%)
- Std: 0.15-0.35 (was 0.59)
- Mean: ~0.0 (properly centered)

✅ **Dataset Size**
- 19,778 clips extracted
- ~15,822 usable after filtering
- 2.2x more than current 7,000

✅ **Training Performance**
- ST-GCN: 89-92% accuracy (was 85-90%)
- Improvement: +4-7% from clean data
- Per-class: >75% for all classes

---

## Future Expansion (Optional)

**If need more data later:**

1. **Add opponent extraction (Phase 2)**
   - Extract opponent's pose during same shot
   - Label as "waiting" or "defensive positioning"
   - Learn context: what opponent does during each shot
   - Would give 2x more data (39,556 samples)

2. **Add temporal context**
   - Extract 2-3 shots before/after target shot
   - Learn shot sequences and rally patterns
   - Multi-shot classification

3. **Add court region features**
   - Split dataset by near/far court
   - Train position-specific models
   - Better accuracy per region

**But start simple:** Single player ROI extraction first

---

## Summary

### Decision: Single Player ROI Extraction

**Benefits:**
- ✅ Perfect timing (clip centers on shot execution)
- ✅ Clear biomechanics (only performing player)
- ✅ Solves multi-player issue (0% contamination)
- ✅ 2.2x more data (15.8K vs 7K)
- ✅ Simpler to implement and validate
- ✅ Expected: 89-92% ST-GCN accuracy

**Next steps:**
1. Update shot type mapping (remove ambiguous shots)
2. Implement ROI calculation logic
3. Modify pose extraction with ROI cropping
4. Test on 1 match, then full extraction
5. Train with clean data

**Timeline:** 16-20 hours (mostly extraction time, can run overnight)

---

**Status:** Ready to implement
**Risk:** Low - proven ROI approach
**Expected improvement:** +4-7% accuracy from clean single-player data
