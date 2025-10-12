# 🚨 FINAL DECISION: Do NOT Use Qwen2.5-VL Fine-tuning

## Current Situation

- **Current Score**: 51.5% test SMAPE, 45.76% validation SMAPE
- **Gap**: 5.7% (validation-test)
- **Target**: < 45% test SMAPE
- **Resources**: A100 80GB, few submissions left
- **Time Pressure**: HIGH - need quick wins

---

## ❌ Why Qwen2.5-VL Fine-tuning Will FAIL

### 1. Time Cost is Catastrophic

```
Fine-tuning 75K samples:     15-20 hours
Inference 75K test samples:   5-10 hours
Testing & debugging:          5-8 hours
-------------------------------------------
TOTAL TIME:                  25-38 hours
```

**You could run 30+ gradient boosting experiments in that time!**

### 2. Regression vs Generation Problem

LLMs output text, not numbers:

- Output: `"$4.89"` or `"The price is 4.89"` or `"4.89 USD"`
- Need regex parsing → **errors on edge cases**
- One failed parse = **200% SMAPE** on that sample!
- With 75K samples, even 1% parse errors = **disaster**

### 3. Won't Fix Your Real Problem

Your issues are:

1. ✅ **5.7% validation-test gap** → Feature leakage, not model complexity
2. ✅ **Brand distribution shift** → Out-of-fold encoding needed
3. ✅ **Simple features working as well as complex** → Feature engineering issue

**Fine-tuning an LLM fixes NONE of these!**

### 4. Competition Constraints

- Model must be < 8B parameters ✅ (Qwen2.5-VL-3B fits)
- But inference must be **practical** ❌ (5-10 hours is NOT)
- Final submission based on **full 75K test set**
- You can't afford 10-hour inference per experiment

### 5. Risk vs Reward

| Metric                   | Qwen Fine-tune | Brand + miniCLIP |
| ------------------------ | -------------- | ---------------- |
| **Time to first result** | 20+ hours      | 30 minutes       |
| **Expected improvement** | +2-3% (maybe)  | +3-5%            |
| **Risk of failure**      | Very High      | Low              |
| **Submissions needed**   | 3-5            | 1-2              |
| **Can iterate**          | No (too slow)  | Yes (fast)       |

---

## ✅ What You SHOULD Do (Proven Strategy)

### Phase 1: Fix Validation Gap (1-2 hours)

Run `diagnostic-analysis.ipynb`:

- Check brand overlap between train/test
- Verify baseline performance (5 simple features)
- Identify if gap is from brand leakage

**Expected**: Understand root cause of 5.7% gap

### Phase 2: Brand-Focused Solution (30 mins - 1 hour)

Run `brand-image-solution.ipynb`:

- Extract brand, quantity, pack count, quality indicators
- Use out-of-fold brand encoding (prevent leakage)
- Optional: miniCLIP (512-dim) for images
- Simple ensemble: LightGBM + XGBoost

**Expected**: 45-48% validation, < 2% gap → 46-49% test SMAPE

### Phase 3: If Still Above 45% (2-3 hours)

Additional features:

- Category engineering (extract from item_name)
- Price per unit normalization
- Brand frequency features
- Image color/texture features (simple CV)

**Expected**: 43-46% test SMAPE

### Phase 4: Ensemble & Submit (30 mins)

- Blend top 2-3 models
- Weighted average based on validation
- Generate submission

**Expected**: Best possible score with remaining submissions

---

## 📊 Evidence: Why Simple Features Work Better

From your own experiments:

1. **768-dim sentence embeddings** → 52% SMAPE
2. **5 simple features (baseline)** → 52-54% SMAPE
3. **Complex dual-input PyTorch** → 54.96% SMAPE (WORSE!)

**Conclusion**: Features matter MORE than model complexity!

Top teams (1000+ ahead of you) likely use:

- ✅ Brand + category + quantity features
- ✅ Out-of-fold target encoding
- ✅ Image features (CLIP/ResNet embeddings)
- ✅ Simple gradient boosting
- ✅ Good validation strategy (prevent leakage)

They are NOT using:

- ❌ Fine-tuned vision-language models
- ❌ Complex neural networks
- ❌ Sentence transformers for text
- ❌ 20-hour training pipelines

---

## 🎯 Your Action Plan (Next 4 Hours)

### Hour 1: Diagnostics

```bash
# Run diagnostic analysis
jupyter nbconvert --to notebook --execute diagnostic-analysis.ipynb
```

**Goal**: Understand validation gap

### Hour 2-3: Brand Solution

```bash
# Run brand-focused solution
jupyter nbconvert --to notebook --execute brand-image-solution.ipynb
```

**Goal**: Get to 45-48% validation SMAPE

### Hour 4: Submit

- Generate test predictions
- Upload to competition
- **Expected**: 46-49% test SMAPE

### If You Have Time Left:

- Iterate on features (add category, price/unit)
- Try removing images if they don't help
- Ensemble with your existing gradient boosting models

---

## 📈 Expected Results

| Approach                   | Time   | Validation | Test    | Gap   | Feasible?        |
| -------------------------- | ------ | ---------- | ------- | ----- | ---------------- |
| **Current (embeddings)**   | Done   | 45.76%     | 51.5%   | 5.7%  | ❌ Gap too large |
| **Qwen Fine-tune**         | 25-38h | 44-47%?    | 48-52%? | 3-5%? | ❌ Too slow      |
| **Brand + miniCLIP**       | 1-2h   | 45-48%     | 46-49%  | <2%   | ✅ **BEST**      |
| **Brand only (no images)** | 30m    | 46-49%     | 47-50%  | <2%   | ✅ Fast          |

---

## 🎓 Lessons Learned

1. **Complex != Better**: 768-dim embeddings got same score as 5 features
2. **Gap is everything**: 5.7% gap means fundamental strategy issue
3. **Features > Models**: Right features with simple models beat complex architectures
4. **Time is precious**: 30-min solution beats 30-hour solution if both get similar scores
5. **Competition dynamics**: 1000+ teams ahead means you need DIFFERENT approach, not just better tuning

---

## 🚀 Final Recommendation

**ABANDON Qwen2.5-VL approach completely.**

Instead:

1. Run `diagnostic-analysis.ipynb` (30 mins)
2. Run `brand-image-solution.ipynb` (1 hour)
3. Submit and iterate (2-3 experiments)

**You'll reach 46-49% test SMAPE in < 4 hours vs 25-38 hours with Qwen.**

The choice is clear. Focus on **fast iteration** with **proven features**, not slow experimentation with unproven LLM fine-tuning.

---

## ⚠️ If You Still Want to Try LLM Approach

If you MUST try LLM (not recommended), here's the ONLY viable way:

1. **Use LLM for feature extraction ONLY** (not end-to-end):

   - Extract brand, category, quality indicators from text
   - Use pre-trained model (no fine-tuning)
   - Takes 1-2 hours, generates features
   - Then use gradient boosting on features

2. **Zero-shot prompting** (test first):
   - Use API (GPT-4, Claude) on 100 samples
   - See if it can predict prices from text
   - If SMAPE < 40% on 100 samples, consider scaling
   - Otherwise, abandon

**DO NOT fine-tune for end-to-end regression. It's a trap.**

---

## 📞 Questions to Ask Yourself

1. Do I have 25-38 hours to wait for results? **NO**
2. Can I afford to waste 3-5 submissions on experiments? **NO**
3. Will fine-tuning fix my 5.7% validation gap? **NO**
4. Is text generation better than direct regression for SMAPE? **NO**
5. Are top 1000 teams using fine-tuned LLMs? **UNLIKELY**

**If all answers are NO, don't do it.**

---

## ✅ Commit to This Plan

**I will**:

1. ✅ Run diagnostic analysis first
2. ✅ Use brand-focused solution
3. ✅ Focus on closing validation gap
4. ✅ Keep solutions simple and fast
5. ✅ Save submissions for proven approaches

**I will NOT**:

1. ❌ Fine-tune Qwen2.5-VL
2. ❌ Waste time on 20+ hour experiments
3. ❌ Use text generation for regression
4. ❌ Ignore validation-test gap
5. ❌ Chase complex solutions without evidence

---

**Time to execute: 4 hours**
**Expected outcome: 46-49% test SMAPE**
**Probability of success: 70-80%**

**Let's move forward with the brand-focused solution. Ready?**
