# 🎯 COMPLETE RESTART STRATEGY - Amazon ML Challenge 2025

## Current Situation

- **Current Performance**: 51.5% test SMAPE, 45.76% validation SMAPE
- **Gap**: 5.7% (HUGE RED FLAG)
- **Problem**: Over-engineering without addressing fundamentals
- **Insight**: 1000+ teams are beating us - we're missing something obvious

## Root Cause Analysis

### What We're Doing Wrong:

1. ❌ **Ignoring images completely** (problem explicitly mentions visual features!)
2. ❌ **Using sentence embeddings that don't capture price signals**
3. ❌ **Complex models (768-dim embeddings) getting same performance as simple ones**
4. ❌ **5.7% val-test gap suggests distribution shift or data leakage**
5. ❌ **No diagnostic analysis of what features actually matter**

### What We're Doing Right:

1. ✅ Out-of-fold encoding (reduces leakage)
2. ✅ Multiple model ensemble
3. ✅ Feature engineering (but wrong features!)

## The Breakthrough Insight

**If 768-dim sentence embeddings were working, we'd have 40% SMAPE, not 52%.**

This means: **Text embeddings are NOT capturing price-relevant information!**

## New Strategy

### Phase 1: Diagnostic Analysis (30 min)

**Goal**: Understand what actually predicts price

```python
# 1. Distribution Analysis
- Check train vs test: value distribution, brands, units
- Adversarial validation on RAW features (not engineered)
- Find which products are mispriced (outliers in residuals)

# 2. Feature Importance
- Train simple LightGBM on ONLY numerical features
- Train on ONLY text embeddings
- Train on ONLY brand/category features
- Compare: which matters most?

# 3. Validation Strategy
- Current: random 15% split
- Try: stratified by price bins
- Try: stratified by brand
- Try: time-based if dates available
```

### Phase 2: Image Features (1 hour) - CRITICAL!

**Why**: Problem statement emphasizes "textual AND visual features"

**Most top teams are definitely using images!**

```python
# Implementation:
1. Download images (use provided download function)
2. Extract features using ResNet50 (2048-dim)
3. Check correlation with price
4. Add to model

# Quick test:
- Train LightGBM with ONLY image features
- If SMAPE < 60%, images matter!
- If SMAPE < 55%, images matter A LOT!
```

### Phase 3: Better Text Features (1 hour)

**Problem**: Sentence embeddings capture semantics, not price

**Solution**: Extract price-relevant features

```python
# Price-Relevant Features:
1. BRAND (exact match)
   - Extract using regex patterns
   - Fuzzy match common brands
   - Brand frequency (rare = expensive?)

2. QUANTITY (normalized)
   - Extract number + unit
   - Normalize to standard units (all to grams/ml)
   - Log(quantity)

3. PACK COUNT
   - Single vs multi-pack
   - Pack size (2, 4, 6, 12, 24)

4. CATEGORY
   - Food vs Electronics vs Cosmetics vs Clothing
   - Use keyword matching, not embeddings

5. QUALITY INDICATORS
   - organic, premium, deluxe, luxury
   - pure, natural, bio
   - Professional, industrial grade

6. SIZE DESCRIPTORS
   - small, medium, large, XL, XXL
   - mini, regular, jumbo, family size
```

### Phase 4: Simpler Models (30 min)

**Drop complexity, focus on features**

```python
# Just 2 models:
1. LightGBM
   - Fast, interpretable
   - Feature importance analysis
   - Handles missing values

2. CatBoost (if we have categorical features)
   - Native categorical support
   - Robust to overfitting

# Maybe add:
3. Simple 2-layer NN (if image features work)
   - Concat: [numerical, image_features, brand_encoded]
   - Dense(512) -> Dense(256) -> Dense(1)
```

### Phase 5: Fix Validation-Test Gap

**5.7% gap is HUGE - this is the main problem!**

Possible causes:

1. **Target encoding leakage** (even with OOF)

   - Solution: Use only mean encoding with smoothing
   - Or: Use only frequency encoding

2. **Different test distribution**

   - Solution: Check adversarial validation
   - If AUC > 0.7, use domain adaptation

3. **Overfitting to validation set**

   - Solution: Use K-fold CV (5 folds)
   - Report mean ± std

4. **Different brands in test**
   - Solution: Add "brand_is_rare" feature
   - Use fallback to category mean

## Implementation Priority

### IMMEDIATE (Next 2 hours):

**Hour 1: Diagnostic**

1. Run adversarial validation on RAW features
2. Check brand overlap between train/test
3. Analyze feature importance (drop embeddings temporarily)
4. Check if removing target encoding reduces gap

**Hour 2: Images**

1. Download 1000 sample images (train + test)
2. Extract ResNet50 features
3. Train LightGBM with ONLY image features
4. If promising, download all images

### NEXT (Following day):

**Phase 1: Better Features**

- Implement brand extraction (regex + fuzzy)
- Normalize quantities to standard units
- Category classification (keyword-based)

**Phase 2: Simpler Pipeline**

- Drop sentence embeddings (use only if they beat baseline)
- Use only: [numerical, brand, category, image]
- 2-model ensemble: LightGBM + CatBoost

**Phase 3: Validation**

- 5-fold CV with proper stratification
- Check if val-test gap reduces to < 2%

## Success Metrics

### Before Submission:

- ✅ Validation SMAPE < 45%
- ✅ Val-Test gap < 2% (CRITICAL!)
- ✅ Feature importance makes sense (brand, quantity, category at top)
- ✅ Images contribute (if not, something's wrong)

### Expected Performance:

- **Conservative**: 45-47% test SMAPE
- **Optimistic**: 42-45% test SMAPE
- **Dream**: 40-42% test SMAPE (probably need images working well)

## Red Flags to Watch:

1. ❌ If removing embeddings doesn't hurt performance → embeddings useless
2. ❌ If images don't correlate with price → wrong extraction method
3. ❌ If val-test gap stays > 3% → fundamental validation strategy issue
4. ❌ If all features have similar importance → not capturing signal

## Key Insights:

1. **Simplicity > Complexity** when features are right
2. **Images are probably THE differentiator** for top teams
3. **Validation-test gap is the main enemy**, not the model
4. **Price prediction needs price-relevant features**, not semantic understanding

---

**Next Step**: Run diagnostic notebook to verify these hypotheses!
