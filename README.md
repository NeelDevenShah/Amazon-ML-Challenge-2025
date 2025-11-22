# 🛒 Amazon ML Challenge 2025 - Price Prediction

This repository contains various approaches for the Amazon ML Challenge 2025 price prediction task.

## 📁 Project Structure

### 📊 [1. Exploratory Data Analysis](./1.%20Exploratory%20Data%20Analysis/)
- Data exploration and feature analysis
- Diagnostic notebooks to understand what drives pricing
- Feature engineering experiments
- **3 notebooks** focused on understanding the dataset

### 🖼️ [2. MultiModal Approach](./2.%20MultiModal%20Approach/)
- Models using both **text** and **images**
- Vision-language models (Qwen2.5, CLIP-based)
- Brand + image feature combinations
- **2 notebooks** leveraging visual product information

### 📝 [3. Text Only Approach](./3.%20Text%20Only%20Approach/)
- Models using **only text** (catalog_content)
- Wide range from traditional ML to modern LLMs
- **30+ implementations** including:
  - Large Language Models (Granite, Qwen, Llama)
  - Encoder Models (T5, FLAN-T5, BERT)
  - Traditional ML (Gradient Boosting, Feature Engineering)
  - Similarity-based methods (FAISS)

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
- Individual notebooks are prefixed with team member names (harsh, neel, sneh)
- Collaborative approaches are organized by methodology

## 🏆 Best Performing Approaches
- Check individual notebook documentation for performance metrics
- Compare validation scores and training times across approaches
- Consider computational requirements for your setup

---
*Organized by approach type for easy navigation and experimentation*
