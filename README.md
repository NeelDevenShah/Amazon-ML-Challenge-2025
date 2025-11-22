# 🛒 Amazon ML Challenge 2025 - Price Prediction

This repository contains various approaches for the Amazon ML Challenge 2025 price prediction task.

## 🏆 **Achievement: All-India Rank 80 / 23,000+ Teams (Top 0.3%)**

**Team ML Mavericks** achieved an exceptional **All-India Rank of 80** out of approximately **23,000 teams**, placing us in the **top 0.3%** with a final **SMAPE score of 43.28** (winning score: 39.7).

**Team Members:**

- Neel Shah
- Sneh Shah
- Harsh Maheshwari
- Harsh Shah

## 🎯 **Problem Statement: Smart Product Pricing Challenge**

### **Challenge Overview**

In e-commerce, determining the optimal price point for products is crucial for marketplace success and customer satisfaction. The challenge was to develop an ML solution that analyzes product details and predicts the price of the product holistically using **only the provided text and product images** - with **all external price lookups strictly prohibited**.

### **Dataset Description**

- **Training Dataset**: 75,000 products with complete details and prices
- **Test Dataset**: 75,000 products for final evaluation

**Features:**

1. `sample_id`: Unique identifier for each sample
2. `catalog_content`: Text field containing title, product description, and Item Pack Quantity (IPQ) concatenated
3. `image_link`: Public URL for product image download
4. `price`: Target variable (training data only)

### **Evaluation Metric**

**SMAPE (Symmetric Mean Absolute Percentage Error)**

```
SMAPE = (1/n) * Σ |predicted_price - actual_price| / ((|actual_price| + |predicted_price|)/2)
```

- Range: 0% to 200% (lower is better)
- Our Final Score: **43.28%**

### **Key Constraints**

- ⚠️ **STRICTLY PROHIBITED**: External price lookup from internet/databases
- Models limited to MIT/Apache 2.0 License with ≤8B parameters
- Must predict positive float values for all test samples

### **Core Challenge Difficulty**

The main complexity was deriving prices **holistically** using only:

- Product titles and descriptions
- Item pack quantities
- Product images
- **NO external price references allowed**

This made it a true test of **feature engineering** and **multimodal learning** capabilities.

## 📁 Project Structure

### 📊 [1. Exploratory Data Analysis](./1.%20Exploratory%20Data%20Analysis/)

Complete data exploration and diagnostic analysis to understand pricing patterns:

- **`classification-idea.ipynb`** - Price classification and binning strategies  
- **`diagnostic-analysis.ipynb`** - Deep diagnostic analysis of price drivers and validation gaps
- **`exploratory-data-analysis-visualization.ipynb`** - Comprehensive EDA with visualizations
- **`feature-engineering.ipynb`** - Advanced feature extraction and preprocessing techniques

**Focus**: Understanding what truly drives product pricing through statistical analysis

### 🖼️ [2. MultiModal Approach](./2.%20MultiModal%20Approach/)

Advanced models leveraging both text and visual information:

- **`multimodal-brand-clip-image-price-prediction.ipynb`** - CLIP-based image + brand features with ensemble
- **`qwen2-5-finetune-multimodal.ipynb`** - Qwen2.5 vision-language model fine-tuning

**Focus**: Combining visual product information with textual descriptions for holistic price prediction

### 📝 [3. Text Only Approach](./3.%20Text%20Only%20Approach/)

Comprehensive text-only solutions spanning traditional ML to cutting-edge LLMs:

#### 🏆 **Final Solutions** (Top Performance)
- **`hybrid-ensemble-validation-test-gap-fixes.ipynb`** - **🥇 FINAL SOLUTION** with validation-test gap fixes
- **`advanced-hybrid-solution.ipynb`** - Ultra-advanced ensemble approach (SMAPE: 38-44%)

#### 🤖 **Large Language Models**
- **`granite-4.0-llm-price-prediction-with-unsloth.ipynb`** - Granite 4.0 with Unsloth optimization
- **`final-granite-amazon-25-alternative.ipynb`** - Alternative Granite implementation
- **`qwen-optimized-fast-training.ipynb`** - Optimized Qwen training pipeline
- **`qwen2-5-finetune-text-only.ipynb`** - Qwen2.5 text-only fine-tuning

#### 🔤 **Encoder Models (T5/FLAN-T5/BERT)**
- **`flan-t5-model-main-third-method-dynamic-length.ipynb`** - FLAN-T5 with dynamic length handling
- **`flan-t5-model-main-inference.ipynb`** - FLAN-T5 inference pipeline
- **`flan-t5-mlp-regression-log-transformed.ipynb`** - T5 + MLP with log transformation
- **`bert-regression-model-price-prediction.ipynb`** - BERT-based regression approach
- **`modern-bert-mmd-loss-price-prediction.ipynb`** - Modern BERT with MMD loss
- **`comprehensive-bert-text-preprocessing-model.ipynb`** - BERT with advanced preprocessing
- **`text-only-bert-optimized-approach.ipynb`** - Optimized BERT implementation

#### 🔧 **Feature Extraction & Engineering**
- **`llm-batch-feature-extraction-15-fields.ipynb`** - LLM-based comprehensive feature extraction
- **`vllm-ultra-fast-feature-extraction-a100.ipynb`** - Ultra-fast GPU-optimized feature extraction
- **`llm-feature-extraction.ipynb`** - General LLM feature extraction pipeline
- **`ml-feature-engineering-approach.ipynb`** - Traditional ML with engineered features

#### 📊 **Traditional ML & Ensemble Methods**
- **`gradient-boosting-solution-amazon-ml.ipynb`** - Gradient boosting implementation
- **`faiss-similarity-search.ipynb`** - FAISS-based similarity search
- **`amazon-ml-price-prediction.ipynb`** - General ML approach

#### 🧪 **Specialized Approaches**
- **`multi-task-t5-beam-search-learning.ipynb`** - Multi-task T5 with beam search
- **`t5-conditional-generation-pytorch-lightning.ipynb`** - T5 with PyTorch Lightning
- **`t5-conditional-generation-pytorch-lightning-alt.ipynb`** - Alternative T5 PyTorch approach
- **`t5-encoder-neural-network-price-classification.ipynb`** - T5 encoder + neural network
- **`tensorflow-lstm-price-prediction-model.ipynb`** - LSTM-based neural network
- **`updated-t5-model-aug-data.ipynb`** - T5 with data augmentation

**Total**: 30+ implementations covering the full spectrum from traditional ML to state-of-the-art LLMs

## 📄 Documentation

- **Amazon 25 Problem Statement.pdf**: Official challenge documentation
- **Multimodal Model Architectures.pdf**: Reference material for multimodal approaches

## 🎯 Challenge Overview

**Task**: Predict product prices from catalog content and images  
**Metric**: SMAPE (Symmetric Mean Absolute Percentage Error)  
**Data**: Product catalog descriptions and images from Amazon

## 🚀 Quick Start

1. **Explore the data**: Start with `1. Exploratory Data Analysis/`
2. **Choose approach**:
   - For image+text: `2. MultiModal Approach/`
   - For text-only: `3. Text Only Approach/`
3. **Check README files** in each folder for detailed descriptions

## 👥 Team Contributions

- Individual notebooks are prefixed with team member names (Neel Shah, Sneh Shah, Harsh Maheshwari, Harsh Shah)
- Collaborative approaches are organized by methodology

## 🏆 Best Performing Approaches

- Check individual notebook documentation for performance metrics
- Compare validation scores and training times across approaches
- Consider computational requirements for your setup

## 🔑 **Key Learnings & Success Factors**

### **1. Data Understanding Before Modeling (80/20 Rule)**

- Biggest gains came from deeply understanding the data before training
- Focus on rigorous preprocessing and thoughtful feature engineering
- Comprehensive exploratory data analysis was crucial

### **2. Robust Validation Strategy**

- With 75k test samples, local validation was critical
- Helped navigate leaderboard and avoid overfitting
- Trust your validation over leaderboard fluctuations

### **3. Team Synergy > Individual Skill**

- Collaborative debugging, pivoting, and motivation
- Diverse approaches and perspectives
- 72-hour intensive sprint requiring sustained teamwork

### **4. Multimodal Feature Engineering**

- Combined text and image features effectively
- Explored vision-language models (Qwen2.5, CLIP-based)
- Brand + image feature combinations proved valuable

### **5. Model Diversity**

- **30+ text-only implementations** from traditional ML to modern LLMs
- Range from Gradient Boosting to Large Language Models
- Ensemble methods combining different model types

## 📈 **Competition Timeline**

- **Duration**: 3-day intensive sprint (72 hours)
- **Public Leaderboard**: Based on 25k test samples
- **Final Ranking**: Complete 75k test set evaluation
- **Final Achievement**: Rank 80/23,000+ teams with SMAPE 43.28%

## 🎯 **Notebook Naming Convention**

All notebooks follow a clear, descriptive naming pattern:
- **`[model/approach]-[specific-technique]-[use-case].ipynb`**
- Example: `hybrid-ensemble-validation-test-gap-fixes.ipynb`
- **No more cryptic names** - every notebook clearly describes its purpose

## 🚀 **Getting Started**

### **For Competition Performance**
1. Start with `3. Text Only Approach/hybrid-ensemble-validation-test-gap-fixes.ipynb` (Final Solution)
2. Compare with `3. Text Only Approach/advanced-hybrid-solution.ipynb` (Ultra-advanced)

### **For Learning & Experimentation**  
1. Begin with `1. Exploratory Data Analysis/` to understand the data
2. Explore different approaches in `3. Text Only Approach/`
3. Try multimodal approaches in `2. MultiModal Approach/`

### **For Specific Model Types**
- **LLMs**: Granite, Qwen notebooks in Text Only folder
- **Traditional ML**: Gradient boosting and feature engineering notebooks  
- **Deep Learning**: BERT, T5, LSTM implementations
- **Multimodal**: CLIP and Qwen2.5 in MultiModal folder

## 📚 **Complete Documentation**

Every notebook now includes:
- ✅ **Professional explanatory markdown** at the beginning
- ✅ **Architecture and approach description**
- ✅ **Key features and innovations**
- ✅ **Expected performance metrics**
- ✅ **Clear, descriptive filenames**

**Total**: 35+ fully documented notebooks across all approaches!
