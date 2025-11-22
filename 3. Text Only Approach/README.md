# 📝 Text Only Approach

This folder contains **30+ models** that use **only text** inputs (catalog_content) for price prediction, spanning from traditional ML to cutting-edge LLMs.

## 🏆 **Final Competition Solutions**

### 🥇 **`hybrid-ensemble-validation-test-gap-fixes.ipynb`** 
- **THE WINNING SOLUTION** - Final competition submission (SMAPE: 40-43%)
- **Architecture**: Hybrid ensemble with adversarial validation and gap fixes
- **Key Innovations**: Separate feature scaling, stronger embeddings, residual connections
- **Impact**: Achieved All-India Rank 80/23,000+ teams

### 🚀 **`advanced-hybrid-solution.ipynb`**
- **Ultra-advanced** ensemble approach (Target SMAPE: 38-44%)
- **Architecture**: 4-model ensemble (LightGBM + CatBoost + XGBoost + Neural Network)
- **Features**: Sentence transformers, advanced target encoding, price clustering
- **Performance**: Top 10-100 leaderboard caliber

## 🤖 **Large Language Model Approaches**

### 💎 **Granite Models**
- **`granite-4.0-llm-price-prediction-with-unsloth.ipynb`** - Granite 4.0 with Unsloth optimization
- **`final-granite-amazon-25.ipynb`** - Main Granite implementation  
- **`final-granite-amazon-25-alternative.ipynb`** - Alternative Granite approach

### 🔮 **Qwen Models**  
- **`qwen2-5-finetune-text-only.ipynb`** - Qwen2.5 text-only fine-tuning
- **`qwen-optimized-fast-training.ipynb`** - Optimized Qwen training pipeline

## 🔤 **Encoder Model Approaches**

### 🤖 **T5/FLAN-T5 Family**
- **`flan-t5-model-main-third-method-dynamic-length.ipynb`** - FLAN-T5 with dynamic length handling
- **`flan-t5-model-main-inference.ipynb`** - FLAN-T5 inference pipeline  
- **`flan-t5-mlp-regression-log-transformed.ipynb`** - T5 + MLP with log transformation
- **`t5-conditional-generation-pytorch-lightning.ipynb`** - T5 with PyTorch Lightning
- **`t5-conditional-generation-pytorch-lightning-alt.ipynb`** - Alternative T5 PyTorch approach
- **`t5-encoder-neural-network-price-classification.ipynb`** - T5 encoder + neural network
- **`updated-t5-model-aug-data.ipynb`** - T5 with data augmentation

### 🧠 **BERT Family**
- **`bert-regression-model-price-prediction.ipynb`** - BERT-based regression
- **`modern-bert-mmd-loss-price-prediction.ipynb`** - Modern BERT with MMD loss  
- **`comprehensive-bert-text-preprocessing-model.ipynb`** - BERT with advanced preprocessing
- **`text-only-bert-optimized-approach.ipynb`** - Optimized BERT implementation

## 🔧 **Feature Extraction & Engineering**

### 🤖 **LLM-Based Feature Extraction**
- **`llm-batch-feature-extraction-15-fields.ipynb`** - Comprehensive 15-field feature extraction
- **`vllm-ultra-fast-feature-extraction-a100.ipynb`** - Ultra-fast GPU-optimized extraction (A100)
- **`llm-feature-extraction.ipynb`** - General LLM feature extraction pipeline

### 🛠️ **Traditional Feature Engineering**
- **`ml-feature-engineering-approach.ipynb`** - Traditional ML with engineered features
- **`faiss-similarity-search.ipynb`** - FAISS-based similarity search

## 📊 **Traditional ML & Specialized Approaches**

### 🌳 **Gradient Boosting**
- **`gradient-boosting-solution-amazon-ml.ipynb`** - Gradient boosting implementation

### 🧪 **Specialized Techniques**
- **`multi-task-t5-beam-search-learning.ipynb`** - Multi-task T5 with beam search
- **`tensorflow-lstm-price-prediction-model.ipynb`** - LSTM-based neural network
- **`amazon-ml-price-prediction.ipynb`** - General ML approach

## 📈 **Performance Tiers**

### 🏆 **Tier 1: Competition Winners (SMAPE < 45%)**
- `hybrid-ensemble-validation-test-gap-fixes.ipynb` - **Final Solution**
- `advanced-hybrid-solution.ipynb` - Ultra-advanced ensemble

### 🥈 **Tier 2: Strong Performers (SMAPE 45-55%)**
- Granite and Qwen LLMs with fine-tuning
- Modern BERT with advanced preprocessing
- FLAN-T5 with dynamic length handling

### 🥉 **Tier 3: Experimental Approaches (SMAPE 55%+)**
- Traditional ML baselines
- Simple encoder models
- Feature extraction pipelines

## 🔧 **Model Categories**

### 🚀 **Large Language Models (7B+ Parameters)**
- **Granite 4.0**, **Qwen2.5** - State-of-the-art LLMs
- Fine-tuning with Unsloth optimization
- Direct text-to-price generation

### 🔤 **Encoder Models (100M-1B Parameters)**  
- **T5/FLAN-T5**, **BERT** variants
- Text encoding + regression heads
- Efficient training and inference

### 📊 **Traditional ML (Feature-Based)**
- **LightGBM**, **XGBoost**, **CatBoost**
- Hand-crafted and LLM-extracted features
- Fast training, interpretable results

### 🧪 **Hybrid Approaches**
- **Ensemble methods** combining multiple model types
- **Feature fusion** from different extraction methods
- **Multi-stage pipelines** (extract → encode → predict)

## ⚡ **Key Advantages**

- **🚀 Faster Training**: No image processing overhead
- **💾 Lower Memory**: Text-only inputs require less RAM
- **⚡ Quick Iteration**: Rapid experimentation cycles
- **🎯 Focused Optimization**: All effort on text understanding
- **📱 Production Ready**: Easier deployment without vision components

## 🏁 **Quick Start Guide**

1. **For Competition Results**: Start with `hybrid-ensemble-validation-test-gap-fixes.ipynb`
2. **For Learning**: Explore `advanced-hybrid-solution.ipynb` 
3. **For LLM Experiments**: Try Granite or Qwen notebooks
4. **For Fast Prototyping**: Use traditional ML approaches
