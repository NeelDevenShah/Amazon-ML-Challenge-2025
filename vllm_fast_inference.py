#!/usr/bin/env python3
"""
Fast vLLM Inference for Granite Price Predictor
Run this after training to get predictions in MINUTES instead of 30+ hours!

Usage: python vllm_fast_inference.py
"""

import pandas as pd
import numpy as np
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import time

# Same text cleaning function from training
def clean_text_enhanced(text):
    if pd.isnull(text):
        return ""
    
    text = str(text).strip()
    
    # Extract structured information
    item_name = re.search(r"Item Name:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    brand = re.search(r"Brand:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    color = re.search(r"Color:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    size = re.search(r"Size:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    material = re.search(r"Material:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    model = re.search(r"Model:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    
    bp1 = re.search(r"Bullet Point\s*1:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    bp2 = re.search(r"Bullet Point\s*2:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    bp3 = re.search(r"Bullet Point\s*3:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    bp4 = re.search(r"Bullet Point\s*4:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    bp5 = re.search(r"Bullet Point\s*5:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    
    value = re.search(r"Value:\s*([\d.,]+)", text, re.IGNORECASE)
    unit = re.search(r"Unit:\s*([A-Za-z]+)", text, re.IGNORECASE)
    description = re.search(r"Description:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    
    structured_parts = []
    
    if item_name:
        structured_parts.append(f"Item: {item_name.group(1).strip()}")
    if value and unit:
        structured_parts.append(f"Quantity: {value.group(1).strip()} {unit.group(1).strip()}")
    elif value:
        structured_parts.append(f"Value: {value.group(1).strip()}")
    
    if brand:
        structured_parts.append(f"Brand: {brand.group(1).strip()}")
    if color:
        structured_parts.append(f"Color: {color.group(1).strip()}")
    if size:
        structured_parts.append(f"Size: {size.group(1).strip()}")
    if material:
        structured_parts.append(f"Material: {material.group(1).strip()}")
    if model:
        structured_parts.append(f"Model: {model.group(1).strip()}")
    
    if bp1:
        structured_parts.append(f"Feature 1: {bp1.group(1).strip()}")
    if bp2:
        structured_parts.append(f"Feature 2: {bp2.group(1).strip()}")
    if bp3:
        structured_parts.append(f"Feature 3: {bp3.group(1).strip()}")
    if bp4:
        structured_parts.append(f"Feature 4: {bp4.group(1).strip()}")
    if bp5:
        structured_parts.append(f"Feature 5: {bp5.group(1).strip()}")
    
    if description:
        structured_parts.append(f"Description: {description.group(1).strip()}")
    
    cleaned_text = ". ".join(structured_parts)
    
    full_text_cleaned = text.lower()
    full_text_cleaned = re.sub(r'[^\w\s.,:\-]', ' ', full_text_cleaned)
    full_text_cleaned = re.sub(r'\s+', ' ', full_text_cleaned)
    full_text_cleaned = full_text_cleaned.strip()
    
    if cleaned_text and full_text_cleaned:
        final_text = f"{cleaned_text}. Full Details: {full_text_cleaned}"
    elif cleaned_text:
        final_text = cleaned_text
    else:
        final_text = full_text_cleaned
    
    return final_text

def main():
    print("=" * 80)
    print("🚀 FAST vLLM INFERENCE FOR GRANITE PRICE PREDICTOR")
    print("=" * 80)
    
    start_time = time.time()
    
    # Load model with vLLM
    print("\n📦 Loading model with vLLM...")
    print("   Model path: granite_price_predictor_vllm/")
    
    llm = LLM(
        model="granite_price_predictor_vllm",
        tensor_parallel_size=1,  # Use 2-4 if you have multiple GPUs
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        dtype="float16"
    )
    print("✅ Model loaded!")
    
    # Load test data
    print("\n📂 Loading test data...")
    test_df = pd.read_csv('dataset/test.csv', encoding='latin1')
    print(f"   Test samples: {len(test_df)}")
    print(f"   Columns: {test_df.columns.tolist()}")
    
    # Clean text
    print("\n🧹 Cleaning catalog content...")
    test_df['catalog_content'] = test_df['catalog_content'].apply(clean_text_enhanced)
    print("✅ Text cleaning complete!")
    
    # Create prompts in Granite format
    print("\n📝 Creating prompts with Granite chat template...")
    prompts = []
    for text in test_df['catalog_content']:
        prompt = (
            f"<|start_of_role|>user<|end_of_role|>"
            f"Predict the price for this product: {text}<|end_of_text|>\n"
            f"<|start_of_role|>assistant<|end_of_role|>"
        )
        prompts.append(prompt)
    
    print(f"✅ Created {len(prompts)} prompts")
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=64,
        stop=["<|end_of_text|>", "\n\n"]
    )
    
    # BATCH INFERENCE - This is where the magic happens!
    print("\n" + "=" * 80)
    print("⚡ RUNNING BATCHED INFERENCE WITH vLLM")
    print("=" * 80)
    print(f"   Batch size: Auto (vLLM optimized)")
    print(f"   Max tokens: 64")
    print(f"   Temperature: 0.1")
    print("\nThis should take MINUTES instead of 30+ hours! ⏱️\n")
    
    inference_start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    inference_time = time.time() - inference_start
    
    print(f"\n✅ Inference complete in {inference_time/60:.2f} minutes!")
    print(f"   Speed: {len(prompts)/inference_time:.2f} samples/second")
    
    # Extract prices
    print("\n💰 Extracting prices from predictions...")
    all_predictions = []
    failed_extractions = 0
    
    for i, output in enumerate(tqdm(outputs, desc="Processing")):
        predicted_text = output.outputs[0].text
        
        # Try multiple regex patterns to extract price
        price_match = re.search(
            r'\$(\d+\.?\d*)|price is (\d+\.?\d*)|predicted price is \$?(\d+\.?\d*)', 
            predicted_text, 
            re.IGNORECASE
        )
        
        if price_match:
            price = float(price_match.group(1) or price_match.group(2) or price_match.group(3))
        else:
            # Fallback to median price from training (update this value)
            price = 50.0
            failed_extractions += 1
        
        # Ensure reasonable price range
        price = max(0.01, min(price, 10000.0))
        all_predictions.append(price)
    
    if failed_extractions > 0:
        print(f"⚠️  Warning: {failed_extractions} predictions used fallback price")
    
    # Create submission
    print("\n📊 Creating submission DataFrame...")
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': all_predictions
    })
    
    # Save submission
    output_file = 'submission_granite_vllm.csv'
    submission.to_csv(output_file, index=False)
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("🎉 INFERENCE COMPLETE!")
    print("=" * 80)
    print(f"✅ Submission saved to: {output_file}")
    print(f"   Shape: {submission.shape}")
    print(f"\n📈 Price Statistics:")
    print(submission['price'].describe())
    print(f"\n⏱️  Performance:")
    print(f"   Total time: {total_time/60:.2f} minutes")
    print(f"   Inference time: {inference_time/60:.2f} minutes")
    print(f"   Speed: {len(prompts)/inference_time:.2f} samples/second")
    print(f"\n🚀 {len(prompts)} predictions completed!")
    print("=" * 80)
    
    return submission

if __name__ == "__main__":
    try:
        submission = main()
        print("\n✅ Script completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
