"""
Verification Script: Check if amazon-ml-price-prediction.ipynb matches reference implementation
"""

print("="*80)
print("VERIFICATION: amazon-ml-price-prediction.ipynb Changes")
print("="*80)

checks = []

# Read the notebook file
with open('/home/neel/Downloads/new-notebook/amazon-ml-price-prediction.ipynb', 'r') as f:
    notebook_content = f.read()

# Check 1: No BatchNorm1d in model
if 'nn.BatchNorm1d' not in notebook_content:
    checks.append(("✅ PASS", "BatchNorm1d layers removed from model"))
else:
    checks.append(("❌ FAIL", "BatchNorm1d layers still present in model"))

# Check 2: 2 epochs configured
if "'num_epochs': 2" in notebook_content:
    checks.append(("✅ PASS", "Training epochs set to 2"))
else:
    checks.append(("❌ FAIL", "Training epochs not set to 2"))

# Check 3: Log transform enabled
if "'apply_log_transform': True" in notebook_content:
    checks.append(("✅ PASS", "Log transformation enabled"))
else:
    checks.append(("❌ FAIL", "Log transformation not enabled"))

# Check 4: DistilBERT model
if "'distilbert-base-uncased'" in notebook_content:
    checks.append(("✅ PASS", "Using DistilBERT model"))
else:
    checks.append(("❌ FAIL", "Not using DistilBERT model"))

# Check 5: Correct architecture layers (512, 256, 128)
if 'nn.Linear(hidden_dim, 512)' in notebook_content and \
   'nn.Linear(512, 256)' in notebook_content and \
   'nn.Linear(256, 128)' in notebook_content:
    checks.append(("✅ PASS", "Model architecture correct (512→256→128→1)"))
else:
    checks.append(("❌ FAIL", "Model architecture incorrect"))

# Check 6: SMAPE calculation present
if 'calculate_smape' in notebook_content:
    checks.append(("✅ PASS", "SMAPE metric calculation present"))
else:
    checks.append(("❌ FAIL", "SMAPE metric calculation missing"))

# Check 7: No extra whitespace normalization in preprocessing
# This is tricky - we need to check the preprocessing doesn't have the join-split pattern
if '".join(cleaned.split())' not in notebook_content or \
   notebook_content.count('".join(') < 3:  # Allow some joins but not in preprocess
    checks.append(("✅ PASS", "Text preprocessing matches reference"))
else:
    checks.append(("⚠️  WARN", "May have extra whitespace normalization"))

# Display results
print("\nVerification Results:")
print("-"*80)
for status, message in checks:
    print(f"{status}: {message}")

print("\n" + "="*80)
passed = sum(1 for status, _ in checks if status == "✅ PASS")
total = len(checks)
print(f"OVERALL: {passed}/{total} checks passed")

if passed == total:
    print("🎉 All changes applied successfully!")
    print("The notebook should now produce results similar to the reference.")
else:
    print("⚠️  Some checks failed. Please review the changes.")

print("="*80)
