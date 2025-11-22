# 📝 Text Only Approach

This folder contains models that use **only text** inputs (catalog_content) for price prediction.

## Contents:

### 🏗️ **LLM-Based Approaches**
- **final-granite-amazon-25.ipynb**: Granite model fine-tuning with Unsloth
- **final-granite-amazon-25-alternative.ipynb**: Alternative Granite approach
- **qwen2-5-finetune-text-only.ipynb**: Qwen2.5 text-only fine-tuning (Llama-3.2-3B)
- **llm-for-train.ipynb**: LLM training pipeline
- **llm-for-train-vllm-optimized.ipynb**: Optimized LLM training with vLLM

### 🤖 **T5/FLAN-T5 Models**
- **flan-t5-model-main-final.ipynb**: Main FLAN-T5 implementation
- **flan-t5-model-main-inference.ipynb**: FLAN-T5 inference pipeline
- **flan-t5-model-main-third-method-dynamic-length.ipynb**: Dynamic length approach
- **flan-t5-mlp-regression-log-transformed.ipynb**: T5 + MLP with log transformation
- **flan-t5-experiment-1.py**: T5 experimental approach
- **flan-t5-mlp-regression-log-transformed.py**: Python version of T5+MLP
- **flan-t5-model-main.py**: Main T5 Python implementation

### 🧠 **BERT-Based Models**
- **new-bert-approach-amazon-25-ml.ipynb**: BERT-based price prediction
- **morderb-bert.ipynb**: Modern BERT implementation
- **harshs-notebook-oct-13.ipynb**: T5 + MLP classifier approach

### 🔍 **Feature Engineering & Traditional ML**
- **gradient-boosting-solution-amazon-ml.ipynb**: Gradient boosting approach
- **ml-feature-engineering-approach.ipynb**: Traditional ML with feature engineering
- **faiss-similarity-search.ipynb**: FAISS-based similarity search
- **llm-feature-extraction.ipynb**: LLM for feature extraction

### 👥 **Team Member Notebooks**
- **neels-notebook-oct-13.ipynb**: Neel's experimental approaches
- **snehs-notebook-oct-13.ipynb**: Sneh's model implementations

### 🔬 **Experimental Approaches**
- **amazon_ml_2025.ipynb**: General Amazon ML approach
- **amazon-ml-price-prediction.ipynb**: Price prediction experiments
- **amzon-2025-wg.ipynb**: Working group experiments
- **Granite4_0.ipynb**: Granite 4.0 experiments
- **qwen-optimized-fast-training.ipynb**: Optimized Qwen training
- **updated-t5-model-aug-data.ipynb**: T5 with data augmentation
- **multi-task-bean-based-learning.ipynb**: Multi-task learning approach
- **asfabc.ipynb**: LSTM-based neural network approach

### 🐍 **Python Scripts**
- **t5-main-2-large-model.py**: Large T5 model implementation
- **t5-main-experiment.py**: T5 experimental script
- **updated-t5-model-aug-data.py**: T5 with augmented data
- **vllm_fast_inference.py**: Fast inference with vLLM
- **log-transformed-multi-task-learning.py**: Multi-task with log transformation
- **multi-task-bean-based-learning.py**: Multi-task learning script

## Model Categories:

### 🚀 **Large Language Models (LLMs)**
- Granite, Qwen2.5, Llama models
- Fine-tuning approaches using Unsloth
- Text generation for price prediction

### 🔤 **Encoder Models**
- T5, FLAN-T5, BERT
- Text encoding + classification/regression heads
- Various pooling and aggregation strategies

### 📊 **Traditional ML**
- Gradient Boosting, XGBoost
- Feature engineering approaches
- Similarity-based methods

## Key Features:
- **Input Type**: Text only (catalog_content)
- **Faster Training**: No image processing overhead
- **Lower Resource Requirements**: Compared to multimodal approaches
- **Diverse Strategies**: From traditional ML to modern LLMs
