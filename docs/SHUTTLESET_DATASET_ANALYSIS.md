# ShuttleSet Dataset Analysis

**Date:** 2026-02-01
**Purpose:** Dataset evaluation for Phase 5 multi-shot type expansion
**Location:** `/Volumes/Ext/GenAI/ShuttleSet/CoachAI-Projects/ShuttleSet/`

---

## Dataset Overview

### Structure

```
ShuttleSet/
├── set/
│   ├── match.csv                    # Match-level metadata
│   └── {match_name}/
│       ├── set1.csv                 # Rally-level shot data
│       ├── set2.csv
│       └── set3.csv
```

### Match Metadata (match.csv)

**Fields:**
- `id`: Match identifier
- `video`: Match name (encoded in filename)
- `tournament`: Tournament name
- `round`: Match round (Finals, Semi-finals, Quarter-finals, Group-Stage)
- `year, month, day`: Match date
- `set`: Number of sets in match
- `duration`: Match duration (minutes)
- `winner, loser`: Player names
- `downcourt`: Court orientation (0 or 1)
- `url`: YouTube video URL

**Sample:**
```csv
1,Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals,Fuzhou Open 2019,Finals,2019,11,10,3,83,Kento MOMOTA,CHOU Tien Chen,0,https://www.youtube.com/watch?v=O669aZhH0LI
```

**Total Matches:** 44 professional badminton matches from 2018-2021

### Rally-Level Data (set{N}.csv)

**Fields:**
- `rally`: Rally number within set
- `ball_round`: Shot number within rally
- `time`: Timestamp in video (HH:MM:SS)
- `frame_num`: Frame number in video
- `roundscore_A, roundscore_B`: Current scores
- `player`: Player who hit (A or B)
- `server`: Server indicator (1 or 2)
- **`type`**: Shot type (Chinese characters) ⭐
- `aroundhead`: Around-the-head shot indicator
- `backhand`: Backhand shot indicator (0=forehand, 1=backhand)
- `hit_height`: Hit height category (1=high, 2=low)
- `hit_area, hit_x, hit_y`: Hit location on court
- `landing_height, landing_area, landing_x, landing_y`: Landing location
- `lose_reason, win_reason`: Rally ending reasons
- `getpoint_player`: Point winner
- `flaw`: Error/flaw indicator
- `player_location_area, player_location_x, player_location_y`: Player position
- `opponent_location_area, opponent_location_x, opponent_location_y`: Opponent position
- `db`: Database field

---

## Shot Types Found

### Complete List (19 shot types)

Scanned 20 CSV files (sample), found 19 unique shot types:

| # | Chinese | English | Count | Category | Notes |
|---|---------|---------|-------|----------|-------|
| 1 | 放小球 | Drop Shot (Soft) | 1,236 | Offensive/Deceptive | Gentle drop to net |
| 2 | 挑球 | Lift / Clear (Defensive) | 1,111 | Defensive | High clear to back court |
| 3 | 擋小球 | Block (Net) | 685 | Defensive | Soft block at net |
| 4 | 推球 | Push Shot | 637 | Offensive | Fast push to mid-court |
| 5 | 長球 | Long Shot / Deep Clear | 530 | Defensive | Deep clear to baseline |
| 6 | 發短球 | Short Serve | 526 | Service | Low serve to front court |
| 7 | 殺球 | **Smash** | 403 | **Offensive** | **Power smash** ⭐ |
| 8 | 切球 | **Slice/Cut Drop** | 397 | **Offensive** | **Cutting drop shot** ⭐ |
| 9 | 點扣 | **Steep Smash / Drop Smash** | 369 | **Offensive** | **Steep angled smash** ⭐ |
| 10 | 過度切球 | Overhead Slice / Cross-Court Drop | 307 | Offensive | Cross-court slice |
| 11 | 勾球 | **Cross-Court Net Shot** | 301 | **Deceptive** | **Net cross-court** ⭐ |
| 12 | 未知球種 | Unknown Shot | 159 | N/A | Unlabeled shots |
| 13 | 平球 | **Drive / Flat Shot** | 130 | **Offensive/Defensive** | **Fast flat shot** ⭐ |
| 14 | 後場抽平球 | Rear-Court Drive | 94 | Offensive | Back-court drive |
| 15 | 撲球 | **Rush/Kill (Net)** | 90 | **Offensive** | **Net kill shot** ⭐ |
| 16 | 防守回抽 | Defensive Drive Return | 69 | Defensive | Defensive counter-drive |
| 17 | 防守回挑 | Defensive Lift Return | 54 | Defensive | Defensive lift |
| 18 | 發長球 | Long Serve | 36 | Service | High serve to back |
| 19 | 小平球 | Short Drive | 9 | Offensive | Short flat shot |

**⭐ = High-priority shots for Phase 5 expansion**

---

## Shot Type Categories

### 1. Overhead Shots (Currently Handled)

| Chinese | English | Status | Count | Notes |
|---------|---------|--------|-------|-------|
| 殺球 | **Smash** | ✅ **Phase 1-4** | 403 | Currently trained (4,641 in our dataset) |
| 長球 | **Clear (Long)** | ✅ **Phase 1-4** | 530 | Currently trained (2,662 in our dataset) |
| 挑球 | **Lift/Clear (Defensive)** | 🔄 **Phase 5** | 1,111 | Similar to Clear, defensive |
| 過度切球 | **Overhead Slice** | 🔄 **Phase 5** | 307 | Overhead variant |

### 2. Drop Shots (Phase 5 Priority)

| Chinese | English | Status | Count | Notes |
|---------|---------|--------|-------|-------|
| 放小球 | **Drop Shot (Soft)** | 🔄 **Phase 5** | 1,236 | Most common shot! |
| 切球 | **Slice/Cut Drop** | 🔄 **Phase 5** | 397 | Offensive drop variant |
| 點扣 | **Steep Smash/Drop** | 🔄 **Phase 5** | 369 | Steep angled attack |

**Total Drop variants:** 2,002 shots (most common category!)

### 3. Net Shots (Phase 5 Consideration)

| Chinese | English | Status | Count | Notes |
|---------|---------|--------|-------|-------|
| 擋小球 | **Block (Net)** | 🔄 **Phase 5** | 685 | Defensive net block |
| 撲球 | **Rush/Kill (Net)** | 🔄 **Phase 5** | 90 | Offensive net kill |
| 勾球 | **Cross-Court Net** | 🔄 **Phase 5** | 301 | Deceptive net shot |

**Total Net shots:** 1,076 shots

### 4. Drive/Push Shots (Phase 5 Consideration)

| Chinese | English | Status | Count | Notes |
|---------|---------|--------|-------|-------|
| 推球 | **Push Shot** | 🔄 **Phase 5** | 637 | Fast push to mid-court |
| 平球 | **Drive/Flat** | 🔄 **Phase 5** | 130 | Fast flat shot |
| 後場抽平球 | **Rear-Court Drive** | 🔄 **Phase 5** | 94 | Back-court drive |
| 小平球 | **Short Drive** | 🔄 **Phase 5** | 9 | Short flat shot |

**Total Drive variants:** 870 shots

### 5. Service Shots (Skip in Phase 5)

| Chinese | English | Status | Count | Notes |
|---------|---------|--------|-------|-------|
| 發短球 | Short Serve | ⏭️ **Skip** | 526 | Service shot - out of scope |
| 發長球 | Long Serve | ⏭️ **Skip** | 36 | Service shot - out of scope |

### 6. Defensive Returns (Phase 5 Consideration)

| Chinese | English | Status | Count | Notes |
|---------|---------|--------|-------|-------|
| 防守回抽 | Defensive Drive Return | 🔄 **Phase 5** | 69 | Defensive counter |
| 防守回挑 | Defensive Lift Return | 🔄 **Phase 5** | 54 | Defensive lift |

---

## Mapping to Our Dataset

### Current Dataset (Phases 1-4)

Our extracted clips use simplified English naming:

| Our Label | ShuttleSet Equivalent | ShuttleSet Chinese | Count (Ours) |
|-----------|----------------------|-------------------|--------------|
| **Clear** | 長球 (Long Shot) + 挑球 (Lift) | 長球, 挑球 | 2,662 |
| **Smash** | 殺球 (Smash) | 殺球 | 4,641 |
| **Drop** | 放小球 + 切球 + 點扣 | 放小球, 切球, 點扣 | 3,179 |
| **Lift** | 挑球 (subset) | 挑球 | 573 |

### Phase 5 Mapping Strategy

**Option 1: Detailed Multi-Class (10+ classes)**
- Map each ShuttleSet shot type individually
- Train on 10+ shot categories
- Challenge: Small counts for some types (e.g., 小平球 only 9 shots)

**Option 2: Simplified 4-Class (Recommended)**
Keep current approach:
- **Clear:** 長球 + 挑球 (overhead defensive)
- **Smash:** 殺球 (overhead offensive)
- **Drop:** 放小球 + 切球 + 點扣 + 過度切球 (all drop variants)
- **Drive/Push:** 推球 + 平球 + 後場抽平球 (mid-court attacks)

**Option 3: Hierarchical 6-Class**
- **Overhead Offensive:** Smash (殺球), Steep Smash (點扣)
- **Overhead Defensive:** Clear (長球), Lift (挑球)
- **Drop:** Soft Drop (放小球), Slice Drop (切球), Overhead Slice (過度切球)
- **Net:** Block (擋小球), Rush/Kill (撲球), Cross-Court (勾球)
- **Drive:** Push (推球), Drive (平球), Rear-Court Drive (後場抽平球)
- **Defensive:** Defensive Drive (防守回抽), Defensive Lift (防守回挑)

---

## Translation Reference

### Complete Chinese-to-English Mapping

```python
SHOT_TYPE_TRANSLATION = {
    # Overhead Shots
    '殺球': 'Smash',
    '長球': 'Clear (Long)',
    '挑球': 'Lift / Clear (Defensive)',
    '過度切球': 'Overhead Slice / Cross-Court Drop',

    # Drop Shots
    '放小球': 'Drop Shot (Soft)',
    '切球': 'Slice Drop / Cut Drop',
    '點扣': 'Steep Smash / Drop Smash',

    # Net Shots
    '擋小球': 'Block (Net)',
    '撲球': 'Rush / Kill (Net)',
    '勾球': 'Cross-Court Net Shot',

    # Drive/Push Shots
    '推球': 'Push Shot',
    '平球': 'Drive / Flat Shot',
    '後場抽平球': 'Rear-Court Drive',
    '小平球': 'Short Drive',

    # Defensive Returns
    '防守回抽': 'Defensive Drive Return',
    '防守回挑': 'Defensive Lift Return',

    # Service
    '發短球': 'Short Serve',
    '發長球': 'Long Serve',

    # Unknown
    '未知球種': 'Unknown Shot'
}
```

---

## Recommendations for Phase 5

### Immediate Next Steps (When Ready)

1. **Stick with 4-class approach:**
   - Clear (overhead defensive): 長球 + 挑球
   - Smash (overhead offensive): 殺球
   - Drop (all variants): 放小球 + 切球 + 點扣 + 過度切球
   - Lift (defensive high clear): Subset of 挑球

2. **Video extraction priorities:**
   - **Drop shots:** Most common (2,002 shots in ShuttleSet sample)
   - **Lift shots:** Underrepresented in our dataset (only 573)

3. **Defer complex shot types:**
   - Net shots (擋小球, 撲球, 勾球) - different biomechanics
   - Drive shots (推球, 平球) - mid-court shots, different phases
   - Service shots - out of scope
   - Defensive returns - specialized defensive shots

### Data Quality Considerations

**ShuttleSet Advantages:**
- ✅ Professional players only (consistent technique)
- ✅ Detailed annotations (19 shot types)
- ✅ Court position data (hit_x, hit_y, landing_x, landing_y)
- ✅ Video timestamps (can extract clips programmatically)
- ✅ YouTube URLs (can download original videos)

**ShuttleSet Limitations:**
- ⚠️ Shot type granularity may be too detailed for ML (19 classes)
- ⚠️ Some shot types have low counts (<100 samples)
- ⚠️ Chinese annotations require translation/mapping
- ⚠️ Professional-only data (may not generalize to amateurs)

---

## Phase 5 Execution Plan (Draft)

### Step 1: Video Extraction
1. Download YouTube videos from match.csv URLs
2. Extract clips using timestamps from set{N}.csv files
3. Map Chinese shot types to our 4-class labels
4. Organize into `clear/`, `smash/`, `drop/`, `lift/` folders

### Step 2: Pose Extraction
1. Run `extract_poses_parallel.py` on new clips
2. Verify pose quality (same standards as Phases 1-4)
3. Create metadata with stroke type labels

### Step 3: Feature Engineering
1. Validate v3 features work for Drop and Lift
2. Check if kinetic chain timing differs for defensive shots
3. Adjust SIS (Smash Intent Score) if needed

### Step 4: Model Training
1. Start with 3-class (Clear, Smash, Drop)
2. Validate accuracy >70% before adding Lift
3. Address Lift class imbalance (oversample or class weights)
4. Retrain on full 4-class dataset

### Step 5: Production Integration
1. Update dual-mode system for 4-class classification
2. Test confidence thresholds for each class
3. Deploy to production

---

## Dataset Statistics (Sample Analysis)

**Files Scanned:** 20 CSV files (out of 104 total)

**Shot Distribution in Sample:**
```
放小球 (Drop Soft):        1,236  (19.5%)
挑球 (Lift/Clear):         1,111  (17.5%)
擋小球 (Block Net):          685  (10.8%)
推球 (Push):                 637  (10.0%)
長球 (Clear Long):           530   (8.4%)
發短球 (Short Serve):        526   (8.3%)
殺球 (Smash):                403   (6.3%)  ⭐
切球 (Slice Drop):           397   (6.3%)  ⭐
點扣 (Steep Smash):          369   (5.8%)  ⭐
過度切球 (Overhead Slice):   307   (4.8%)
勾球 (Cross-Court Net):      301   (4.7%)
未知球種 (Unknown):          159   (2.5%)
平球 (Drive):                130   (2.0%)
後場抽平球 (Rear Drive):      94   (1.5%)
撲球 (Rush/Kill):             90   (1.4%)
防守回抽 (Def Drive):         69   (1.1%)
防守回挑 (Def Lift):          54   (0.9%)
發長球 (Long Serve):          36   (0.6%)
小平球 (Short Drive):          9   (0.1%)
```

**Total Shots in Sample:** 6,343 shots from 20 match sets

---

## Next Steps

**For Phase 5 Planning:**
1. ✅ Dataset evaluated and shot types catalogued
2. ✅ Chinese-to-English translations documented
3. ⏳ **Awaiting user instructions** for video extraction
4. ⏳ Script to download YouTube videos and extract clips
5. ⏳ Mapping logic from ShuttleSet labels to our 4-class system

**Dependencies:**
- User provides video extraction requirements/preferences
- Phase 4 (Clear+Smash binary classification) deployed to production
- Binary system validated with real users

---

**Last Updated:** 2026-02-01
**Status:** Ready for Phase 5 planning when Phases 1-4 complete
