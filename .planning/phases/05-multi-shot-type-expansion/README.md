# Phase 5: Multi-Shot Type Expansion

**Status:** Not started (deferred until Phase 4 complete)
**Goal:** Expand from binary (Clear vs Smash) to multi-class classification (Clear, Smash, Drop, Lift)

---

## Overview

This phase adds Drop and Lift shot types to the ML coaching system after the binary Clear vs Smash classification has been validated in production.

## Current State (After Phase 4)

**Trained on:**
- Clear: 2,662 videos (36.4%)
- Smash: 4,641 videos (63.6%)
- **Total:** 7,303 videos

**Classification:** Binary (Clear vs Smash)
**Accuracy Target:** 70%+ test accuracy achieved
**Production:** Dual-mode system (ML + benchmark fallback) deployed

## Phase 5 Scope

**Add to dataset:**
- Drop: 3,179 videos (28.8%)
- Lift: 573 videos (5.2%)

**After Phase 5:**
- **Total:** 11,055 videos
- **Classification:** 4-class (Clear, Smash, Drop, Lift)
- **Accuracy Target:** 65%+ across all 4 classes

## Key Challenges

### 1. Class Imbalance

Distribution after adding Drop and Lift:
- Smash: 4,641 (42.0%) ✓
- Drop: 3,179 (28.8%) ✓
- Clear: 2,662 (24.1%) ✓
- Lift: 573 (5.2%) ⚠️ **Severely underrepresented**

**Mitigation strategies:**
- Option A: Oversample Lift class (SMOTE or simple duplication)
- Option B: Use class weights in training
- Option C: Start with 3-class (Clear, Smash, Drop) and add Lift later
- Option D: Collect more Lift shot videos before training

### 2. Video Extraction

**User will provide:**
- Instructions for extracting Drop and Lift video clips
- Quality standards and annotation requirements
- Timeline for video preparation

**Requirements:**
- Same extraction quality as Clear and Smash
- Consistent pose detection quality
- Proper labeling and metadata

### 3. Feature Engineering Validation

**To verify:**
- Current v3 features work for Drop and Lift
- Kinetic chain timing differs for defensive shots (Drop, Lift)
- SIS (Smash Intent Score) may need adjustment or new metric
- Phase segmentation works for non-overhead shots

### 4. Model Architecture

**Considerations:**
- Binary models (Clear vs Smash) won't transfer directly
- Multi-class requires different output layer and loss function
- May need architecture adjustments for 4-way classification
- Consider hierarchical classification (overhead vs defensive → specific shot)

## Prerequisites

Before starting Phase 5:

- [x] Phase 1 complete (Infrastructure)
- [x] Phase 2 complete (Feature Engineering for Clear+Smash)
- [ ] Phase 3 complete (Models trained on Clear+Smash)
- [ ] Phase 4 complete (Binary system deployed to production)
- [ ] Binary system validated with real users
- [ ] User provides Drop/Lift video extraction instructions
- [ ] Drop and Lift videos extracted and annotated
- [ ] Pose extraction completed for Drop and Lift

## Success Criteria

1. Drop and Lift video clips extracted with same quality standards as Clear and Smash
2. Pose extraction completed for all Drop (3,179) and Lift (573) videos
3. Multi-class models achieve >65% accuracy across all 4 shot types
4. Class imbalance addressed (Lift is only 5.2% - oversampling or class weights applied)
5. Production system seamlessly handles 4-class classification with same confidence thresholds

## Proposed Approach

### Option A: Full 4-Class Expansion

1. Extract Drop and Lift videos (user provides instructions)
2. Run pose extraction on new videos
3. Validate v3 features work for all shot types
4. Address class imbalance (oversample Lift or use class weights)
5. Retrain models on 11,055 videos (4 classes)
6. Evaluate on test set (>65% accuracy target)
7. Deploy to production with 4-class support

**Pros:**
- Complete solution in one phase
- All shot types available immediately

**Cons:**
- Lift severely underrepresented (5.2%)
- More complex deployment
- Higher risk of accuracy drop

### Option B: Incremental 3-Class → 4-Class

**Step 1: Add Drop (3-class)**
1. Extract Drop videos
2. Train on Clear (2,662) + Smash (4,641) + Drop (3,179) = 10,482 videos
3. Validate 3-class accuracy >70%
4. Deploy to production

**Step 2: Add Lift (4-class)**
5. Extract Lift videos
6. Address Lift underrepresentation
7. Train on all 11,055 videos
8. Validate 4-class accuracy >65%
9. Deploy to production

**Pros:**
- Lower risk - validate each addition
- Better class balance for 3-class
- Easier to debug issues

**Cons:**
- Two deployment cycles
- More time to complete

## Recommended Path Forward

**Recommendation:** Option B (Incremental)

**Rationale:**
1. Lift is severely underrepresented (5.2%) - needs special handling
2. Drop has good representation (28.8%) - safe to add first
3. Validates multi-class expansion before tackling imbalance
4. Lower risk approach for production system

## Next Steps (When Ready)

1. **User provides video extraction instructions** for Drop and Lift
2. **Extract and annotate videos** according to requirements
3. **Run pose extraction** using existing pipeline
4. **Validate feature engineering** on new shot types
5. **Plan Phase 5 execution** using `/gsd:plan-phase 5`

## Notes

- This phase is deferred until Phases 1-4 complete successfully
- Focus remains on Clear vs Smash binary classification until production validation
- User will provide specific instructions for Drop/Lift extraction when ready
- Consider collecting additional Lift videos to address class imbalance

---

**Status:** Awaiting Phase 4 completion and user instructions

**Last updated:** 2026-02-01
