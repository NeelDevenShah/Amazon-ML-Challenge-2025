# Fast Inference with vLLM - Quick Start Guide

## Problem
- One-by-one predictions take **30+ hours** ⏰
- This is NOT practical for 75,000 test samples

## Solution: vLLM Batched Inference ⚡
With vLLM on A100, predictions take **MINUTES** instead of hours!

---

## Step-by-Step Instructions

### 1️⃣ Complete Training (in Jupyter/Colab)
Run all cells in `Granite4_0.ipynb` up to and including the model saving cell.

This will create: `granite_price_predictor_vllm/` folder

### 2️⃣ Install vLLM
```bash
# Install vLLM (only needed once)
pip install vllm

# Or with all dependencies
pip install vllm torch transformers
```

### 3️⃣ Run Fast Inference
```bash
# Run the vLLM inference script
python vllm_fast_inference.py
```

That's it! You'll get `submission_granite_vllm.csv` in minutes! 🎉

---

## Performance Comparison

| Method | Time | Speed |
|--------|------|-------|
| One-by-one (original) | **30+ hours** | ~0.7 samples/sec |
| vLLM batched (A100) | **5-10 minutes** | ~150+ samples/sec |

**Speed up: ~200x faster!** 🚀

---

## What the Script Does

1. ✅ Loads your trained model with vLLM
2. ✅ Loads and cleans test data
3. ✅ Creates prompts in Granite format
4. ✅ **Runs BATCHED inference** (key optimization!)
5. ✅ Extracts prices from predictions
6. ✅ Saves submission CSV

---

## Expected Output

```
🚀 FAST vLLM INFERENCE FOR GRANITE PRICE PREDICTOR
================================================

📦 Loading model with vLLM...
✅ Model loaded!

📂 Loading test data...
   Test samples: 75000

⚡ RUNNING BATCHED INFERENCE WITH vLLM
================================================

✅ Inference complete in 8.5 minutes!
   Speed: 147 samples/second

📊 Creating submission DataFrame...
✅ Submission saved to: submission_granite_vllm.csv
```

---

## Troubleshooting

### Error: "Model not found"
- Make sure you ran the training notebook and saved the model
- Check that `granite_price_predictor_vllm/` folder exists

### Error: "CUDA out of memory"
- Reduce `gpu_memory_utilization=0.9` to `0.7` in the script
- Or use `tensor_parallel_size=2` if you have 2 GPUs

### Predictions seem wrong
- Check that text cleaning function matches training
- Verify the price extraction regex patterns
- Look at a few raw model outputs to debug

---

## Advanced Options

### Use Multiple GPUs
```python
llm = LLM(
    model="granite_price_predictor_vllm",
    tensor_parallel_size=2,  # Use 2 GPUs
    ...
)
```

### Adjust Batch Size
vLLM automatically optimizes batch size, but you can tune:
```python
sampling_params = SamplingParams(
    temperature=0.1,
    max_tokens=64,
    # Add more parameters as needed
)
```

---

## Why vLLM is So Fast

1. **PagedAttention**: Efficient KV cache management
2. **Continuous Batching**: Processes multiple requests simultaneously
3. **Optimized CUDA kernels**: Hardware-optimized operations
4. **Dynamic Batching**: Automatically groups requests for efficiency

Instead of:
```
for sample in test_data:  # 75,000 iterations
    predict(sample)        # Each takes ~1.4 seconds
```

vLLM does:
```
outputs = llm.generate(all_prompts)  # Single batched call!
```

---

## Files Created

- `granite_price_predictor_vllm/` - Merged model for vLLM
- `vllm_fast_inference.py` - Fast inference script
- `submission_granite_vllm.csv` - Final predictions

---

## Questions?

Check the vLLM docs: https://docs.vllm.ai/

Good luck with your predictions! 🚀
