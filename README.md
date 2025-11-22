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
- **Final Achievement**: Rank 80/23,000+ teams with SMAPE 43.
