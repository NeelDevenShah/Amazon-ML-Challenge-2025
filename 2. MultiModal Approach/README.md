# 🖼️ MultiModal Approach

This folder contains models that use both **text** and **image** inputs for price prediction.

## Contents:

### 🎯 **brand-image-solution.ipynb**

- Brand + Image solution using miniCLIP for image processing
- Combines text features (brand, quantity, pack count) with image embeddings
- Uses LightGBM + XGBoost ensemble
- Focus on features that actually predict price

### 🤖 **qwen2-5-finetune-multimodal.ipynb**

- Qwen2.5 model fine-tuned for multimodal input
- Uses both catalog content (text) and product images
- Image downloading and preprocessing pipeline
- Vision-language model approach

## Key Features

- **Input Types**: Text (catalog_content) + Images (image_link)
- **Models**: Vision-Language Models, CLIP-based approaches
- **Strategy**: Leverage visual information alongside textual product descriptions
- **Expected Performance**: Better understanding of products through visual context

## Note

These approaches require image processing capabilities and typically need more computational resources due to the multimodal nature.
