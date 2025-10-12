# 🚀 Qwen2.5-VL Optimized Training Guide

## ⚠️ READ THIS FIRST

**Total Time Commitment: 10-16 hours on A100 80GB**

Before you start, understand what you're committing to:

- ✅ Training: 8-12 hours
- ✅ Validation: 30 minutes
- ✅ Test inference: 2-4 hours (with vLLM)
- ✅ **Total: 10-16 hours minimum**

**You cannot iterate quickly with this approach.**

---

## 📋 What's Optimized in `qwen-optimized-fast-training.ipynb`

### 1. **Training Speed (2x faster)**

- ✅ Unsloth for 2x training speedup
- ✅ Larger batch size (8 vs 2) on A100
- ✅ Gradient accumulation (effective batch: 16)
- ✅ Mixed precision (FP16)
- ✅ 8-bit AdamW optimizer
- ✅ Optimized gradient checkpointing

**Result**: 8-12 hours instead of 15-20 hours

### 2. **Inference Speed (5-10x faster)**

- ✅ vLLM for continuous batching
- ✅ PagedAttention for memory efficiency
- ✅ Batch size 1000 (vLLM handles internally)
- ✅ FP16 precision

**Result**: 2-4 hours instead of 8-10 hours

### 3. **Prompt Engineering**

```
You are a price prediction expert. Predict the product price in USD based on the catalog description and image.

CRITICAL RULES:
1. Output ONLY a numeric price (e.g., 12.99)
2. NO dollar signs, NO text, NO explanations
3. DO NOT use quantity/weight numbers as price (e.g., "12 oz" is NOT $12)
4. Consider: brand quality, product type, packaging, quantity
5. Typical range: $0.50 to $500.00 for most products

Output format: Just the number
Example: 14.99
```

**Why this works:**

- Clear anti-hallucination instructions
- Prevents confusing quantity with price
- Enforces numeric-only output
- Gives realistic price range context

### 4. **Robust Price Parsing**

```python
def parse_price_output(text):
    """Handles multiple output formats gracefully."""
    # Removes: $, USD, text
    # Extracts: first numeric value
    # Validates: 0.01 to 10000 range
    # Fallback: Returns None (not crash)
```

**Handles:**

- "12.99" ✅
- "$12.99" ✅
- "The price is 12.99" ✅
- "12.99 USD" ✅
- Garbage output → Fallback to mean price ✅

### 5. **Training Strategy**

- 80/20 train/validation split
- 2 epochs (enough for 75K samples)
- Low temperature (0.1) for consistent output
- Evaluation every 500 steps
- Checkpoint every 1000 steps
- Keep only 2 checkpoints (save space)

---

## 📊 Expected Performance

| Metric         | Optimized       | Original        | Savings  |
| -------------- | --------------- | --------------- | -------- |
| **Training**   | 8-12 hours      | 15-20 hours     | **40%**  |
| **Inference**  | 2-4 hours       | 8-10 hours      | **70%**  |
| **Total**      | **10-16 hours** | **23-30 hours** | **45%**  |
| **GPU Memory** | 50-60 GB        | 60-70 GB        | 10-15 GB |

---

## 🎯 Step-by-Step Execution

### Phase 1: Setup (5 minutes)

1. **Upload notebook to Kaggle/Colab with A100**

   ```bash
   # Make sure you have A100 80GB GPU!
   # Check: GPU Settings → A100
   ```

2. **Configure paths in notebook**

   ```python
   DATASET_FOLDER = '/kaggle/input/amazon-ml-challenge-2025/...'
   USE_IMAGES = True  # Set False for text-only (faster, less accurate)
   SAMPLE_SIZE = None  # None = all 75K, or 1000 for testing
   ```

3. **Run Step 1-2: Install & Config** (5 mins)

### Phase 2: Data Prep (30-60 minutes)

4. **Run Step 3: Load Data**

   - Loads 75K train, 75K test
   - Splits 80/20 train/val

5. **Run Step 4: Download Images** (30-60 mins)

   - Downloads 150K images total
   - Uses 100 parallel workers
   - **⚠️ SKIP if USE_IMAGES=False**

6. **Run Step 5-6: Prepare Dataset** (5-10 mins)
   - Converts to conversation format
   - Loads images into memory

### Phase 3: Training (8-12 hours)

7. **Run Step 7-8: Load Model & LoRA**

   - Loads Qwen2.5-VL-3B in 4-bit
   - Adds LoRA adapters

8. **Run Step 9: Test Before Training**

   - Verify model works
   - See baseline prediction

9. **Run Step 10: TRAIN** (8-12 hours)
   - **⚠️ CRITICAL: Don't interrupt!**
   - Monitor GPU: `watch -n 1 nvidia-smi`
   - Expected: 50-60 GB GPU usage
   - **Go do something else for 10 hours**

### Phase 4: Validation (30 minutes)

10. **Run Step 11: Save Model**

    - Saves LoRA adapters
    - Merges for vLLM (5-10 mins)

11. **Run Step 12-13: Validation**
    - Tests on validation set
    - Calculates SMAPE
    - **CRITICAL**: If SMAPE > 50%, STOP and reconsider

### Phase 5: Test Inference (2-4 hours)

12. **Run Step 14-15: vLLM Inference**

    - Loads with vLLM
    - Generates 75K predictions (2-4 hours)
    - Creates submission.csv

13. **Submit to competition**

---

## 🚨 Critical Checkpoints

### Checkpoint 1: After Data Prep

**Check:**

- ✅ Images downloaded (if USE_IMAGES=True)
- ✅ Train: 60K samples, Val: 15K samples
- ✅ Conversation format looks correct

**If not:** Fix data issues before training

### Checkpoint 2: After 1 Hour of Training

**Check:**

- ✅ GPU usage: 50-60 GB
- ✅ Training loss decreasing
- ✅ No OOM errors
- ✅ Samples/sec: ~1-2

**If not:** Reduce batch size to 4

### Checkpoint 3: After Training Complete

**Check:**

- ✅ Validation SMAPE calculated
- ✅ Model saved

**Decision point:**

- If SMAPE < 45%: ✅ Continue to test inference
- If SMAPE 45-50%: ⚠️ Risky, but proceed
- If SMAPE > 50%: ❌ **STOP - Try brand-focused solution instead**

### Checkpoint 4: After Validation Test (5 samples)

**Check:**

- ✅ Predictions are numbers (not text)
- ✅ Predictions in reasonable range ($0.50-$500)
- ✅ Not predicting quantities as prices

**If not:** Fix prompt or parsing logic

---

## 🐛 Troubleshooting

### Problem: OOM (Out of Memory)

**Solutions:**

```python
# Option 1: Reduce batch size
PER_DEVICE_BATCH_SIZE = 4  # was 8

# Option 2: Reduce gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 1  # was 2

# Option 3: Reduce max length
MAX_LENGTH = 1024  # was 2048
```

### Problem: Training too slow (< 1 sample/sec)

**Solutions:**

```python
# Option 1: Use text-only (no images)
USE_IMAGES = False

# Option 2: Reduce image resolution (modify download function)
# Option 3: Use smaller sample for testing
SAMPLE_SIZE = 10000  # Instead of 75K
```

### Problem: Model outputs text instead of numbers

**Solutions:**

1. Check prompt format (Step 5)
2. Lower temperature to 0.01
3. Add more examples to prompt
4. Check parsing function

### Problem: Predictions all similar (e.g., all ~$15)

**Causes:**

- Model not learning (need more epochs)
- Features not informative (images not helping)
- Overfitting to mean price

**Solutions:**

- Train for 3 epochs instead of 2
- Add more diverse training samples
- Check if brand/category features extracted

### Problem: vLLM fails to load

**Solutions:**

```python
# Option 1: Make sure merged model saved
SAVE_MERGED = True  # In Step 11

# Option 2: Use HuggingFace inference (slower)
# Skip vLLM, use model.generate() in loop

# Option 3: Check vLLM version
!pip install vllm==0.6.0  # Specific version
```

---

## 📈 Performance Optimization Tips

### For Faster Training:

1. **Use text-only** (skip images):

   - Set `USE_IMAGES = False`
   - Reduces training time by 30-40%
   - Accuracy drop: ~2-5% SMAPE

2. **Use smaller sample first**:

   - Set `SAMPLE_SIZE = 5000`
   - Train in 30-60 minutes
   - Validate approach before full run

3. **Reduce epochs**:
   - Set `NUM_EPOCHS = 1`
   - Faster but may underfit

### For Better Accuracy:

1. **Include images**:

   - Set `USE_IMAGES = True`
   - Adds visual signals (packaging, branding)

2. **More epochs**:

   - Set `NUM_EPOCHS = 3`
   - But watch for overfitting

3. **Larger LoRA rank**:
   ```python
   LORA_R = 32  # was 16
   LORA_ALPHA = 32
   ```
   - More parameters, better capacity

---

## 🎯 Success Criteria

### Minimum Viable:

- ✅ Training completes without OOM
- ✅ Validation SMAPE < 52%
- ✅ Test predictions are numbers (not text)
- ✅ Parsing success rate > 95%

### Good Result:

- ✅ Validation SMAPE: 45-50%
- ✅ Test SMAPE: 46-52%
- ✅ Gap < 5%

### Excellent Result:

- ✅ Validation SMAPE: < 45%
- ✅ Test SMAPE: < 47%
- ✅ Gap < 2%
- ✅ **Competitive with top 1000 teams**

---

## ⚖️ Reality Check

### What This Approach Is Good For:

- ✅ Learning end-to-end multimodal fine-tuning
- ✅ Leveraging both text AND images
- ✅ Understanding vision-language models
- ✅ If you have 12-16 hours to spare

### What This Approach Is NOT Good For:

- ❌ Quick iteration (10+ hours per experiment)
- ❌ Few submissions remaining (can't experiment)
- ❌ Fixing validation-test gap (doesn't address leakage)
- ❌ Learning from errors (too slow to debug)

### Better Alternatives (if time-constrained):

1. **Brand-focused + miniCLIP** (1-2 hours total)

   - Extract brand, quantity, category
   - Pre-computed CLIP embeddings
   - LightGBM/XGBoost ensemble
   - Expected: 46-49% test SMAPE

2. **Zero-shot with GPT-4** (test first!)
   - Test on 100 samples (~$0.50 cost)
   - If < 40% SMAPE, scale up
   - Otherwise, don't waste money

---

## 📊 Comparison Table

| Approach                 | Time   | Val SMAPE | Test SMAPE | Iterations | Risk |
| ------------------------ | ------ | --------- | ---------- | ---------- | ---- |
| **Qwen Fine-tune**       | 12-16h | 44-50%?   | 46-52%?    | 1-2 max    | High |
| **Brand + miniCLIP**     | 1-2h   | 45-48%    | 46-49%     | 5-10       | Low  |
| **Current (embeddings)** | Done   | 45.76%    | 51.5%      | N/A        | N/A  |

---

## ✅ Final Checklist Before Starting

- [ ] I have A100 80GB GPU available
- [ ] I have 12-16 hours of uninterrupted time
- [ ] I understand I can only do 1-2 experiments
- [ ] I've set realistic expectations (46-52% test SMAPE)
- [ ] I have a backup plan if this doesn't work
- [ ] I've read the troubleshooting section
- [ ] I understand vLLM inference takes 2-4 hours

**If all checked:** Proceed with `qwen-optimized-fast-training.ipynb`

**If any unchecked:** Reconsider `brand-image-solution.ipynb` instead

---

## 🆘 Emergency Exit Plan

If things go wrong:

1. **After 2 hours of training:**

   - Check loss is decreasing
   - If not, STOP and debug

2. **After validation SMAPE > 52%:**

   - **ABORT mission**
   - Don't waste time on test inference
   - Switch to brand-focused solution

3. **After test inference shows garbage:**
   - Check parsing function
   - Manually inspect 10-20 predictions
   - Fix and re-run (another 2-4 hours)

---

## 📞 Support

If you encounter issues:

1. Check troubleshooting section above
2. Review Unsloth docs: https://docs.unsloth.ai/
3. Review vLLM docs: https://docs.vllm.ai/

**Remember:** You're trading time for potential accuracy. Make sure it's worth it!

---

**Last Updated:** Based on FINAL_DECISION.md analysis
**Status:** Optimized but still high-risk/high-time approach
**Recommendation:** Consider alternatives unless you have time to spare

Good luck! 🚀
