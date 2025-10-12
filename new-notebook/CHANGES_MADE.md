# Changes Made to amazon-ml-price-prediction.ipynb

## Summary

Fixed the notebook to match the reference implementation (amazon-ml-2025.ipynb) to achieve better loss and SMAPE scores.

## Critical Changes Made:

### 1. Model Architecture - Removed BatchNorm Layers ✅

**Location:** `TransformerPriceRegressor` class (Section 8)

**Before:**

```python
self.price_regressor = nn.Sequential(
    nn.Linear(hidden_dim, 512),
    nn.ReLU(),
    nn.Dropout(dropout_prob),
    nn.BatchNorm1d(512),  # ❌ REMOVED

    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(dropout_prob),
    nn.BatchNorm1d(256),  # ❌ REMOVED

    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(dropout_prob),
    nn.BatchNorm1d(128),  # ❌ REMOVED

    nn.Linear(128, 1)
)
```

**After:**

```python
self.price_regressor = nn.Sequential(
    nn.Linear(hidden_dim, 512),
    nn.ReLU(),
    nn.Dropout(dropout_prob),

    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(dropout_prob),

    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(dropout_prob),

    nn.Linear(128, 1)
)
```

**Why:** BatchNorm1d can cause issues with:

- Variable batch sizes (especially last batch)
- Training vs evaluation mode discrepancies
- Instability with regression tasks

### 2. Training Epochs - Changed from 3 to 2 ✅

**Location:** `HYPERPARAMS` dictionary (Section 5)

**Before:** `'num_epochs': 3`
**After:** `'num_epochs': 2`

**Why:** Reference notebook trains for exactly 2 epochs, which was tuned for optimal performance.

### 3. Text Preprocessing - Removed Extra Whitespace Normalization ✅

**Location:** `preprocess_text` function (Section 2)

**Before:**

```python
cleaned = re.sub(r"[^\w\s]", "", cleaned)
cleaned = " ".join(cleaned.split())  # ❌ REMOVED
return cleaned
```

**After:**

```python
cleaned = re.sub(r"[^\w\s]", "", cleaned)
return cleaned
```

**Why:** Exact match with reference preprocessing to ensure identical tokenization behavior.

## Expected Results:

With these changes, the model should now achieve:

- **Lower Loss:** Similar to reference implementation
- **Lower SMAPE:** Matching or close to reference SMAPE scores
- **More Stable Training:** No BatchNorm-related issues

## What Remained the Same (Correct Implementation):

✅ Log transformation of prices (`apply_log_transform: True`)
✅ Outlier removal using IQR method
✅ Model architecture (512 → 256 → 128 → 1)
✅ DistilBERT base model
✅ Learning rate, dropout, batch size, max length
✅ SMAPE calculation
✅ Training and validation split (20%)

## Notes:

The main issue was the **BatchNorm1d layers** which can cause:

1. Numerical instability during training
2. Different behavior between training and evaluation modes
3. Issues with small batch sizes or the last batch

Removing these layers makes the model architecture identical to the reference and should result in the same performance.
