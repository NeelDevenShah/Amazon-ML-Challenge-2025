# 🖼️ MultiModal Approach

This folder contains models that use both **text** and **image** inputs for price prediction.

## Contents:

### 🎯 **`multimodal-brand-clip-image-price-prediction.ipynb`**

- **Purpose**: Advanced multimodal solution combining visual and textual information
- **Architecture**: CLIP-based image encoding + text feature extraction + ensemble ML
- **Key Features**: 
  - Brand extraction and encoding
  - Product image processing with CLIP embeddings
  - Quantity and pack count parsing
  - LightGBM + XGBoost ensemble
- **Performance**: Enhanced accuracy through visual context understanding
- **Techniques**: Computer vision, feature fusion, gradient boosting ensemble

### 🤖 **`qwen2-5-finetune-multimodal.ipynb`**

- **Purpose**: State-of-the-art vision-language model fine-tuning
- **Architecture**: Qwen2.5 multimodal transformer fine-tuned for price prediction
- **Key Features**:
  - End-to-end multimodal learning
  - Image downloading and preprocessing pipeline
  - Joint text-image representation learning
  - Direct price regression from multimodal inputs
- **Performance**: Leverages pre-trained vision-language understanding
- **Techniques**: Transfer learning, multimodal transformers, fine-tuning optimization

## Key Features

- **Input Types**: Text (catalog_content) + Images (image_link)
- **Models**: Vision-Language Models, CLIP-based approaches
- **Strategy**: Leverage visual information alongside textual product descriptions
- **Expected Performance**: Better understanding of products through visual context

## Note

These approaches require image processing capabilities and typically need more computational resources due to the multimodal nature.
