import pandas as pd
import numpy as np
import re
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tqdm.auto import tqdm
import warnings

# --- Configuration ---
warnings.filterwarnings('ignore')
pl.seed_everything(42)  # for reproducibility

MODEL_NAME = 't5-large'
BATCH_SIZE = 32  # Increased for H100. Can be increased further based on memory.
LEARNING_RATE = 1e-4
MAX_EPOCHS = 5
SOURCE_MAX_LEN = 256
TARGET_MAX_LEN = 8

# --- Data Cleaning Function (Not used as per request, but kept for reference) ---
def clean_text_enhanced(text):
    """
    Cleans the catalog content by extracting structured info and performing basic cleaning.
    """
    if pd.isnull(text):
        return ""

    # Extract structured information first
    item_name = re.search(r"Item Name:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    bp1 = re.search(r"Bullet Point\s*1:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    bp2 = re.search(r"Bullet Point\s*2:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    value = re.search(r"Value:\s*([\d.,]+)", text, re.IGNORECASE)
    unit = re.search(r"Unit:\s*([A-Za-z]+)", text, re.IGNORECASE)

    # Build structured text
    structured_parts = []
    if item_name: structured_parts.append(f"Item: {item_name.group(1).strip()}")
    if bp1: structured_parts.append(f"Feature: {bp1.group(1).strip()}")
    if bp2: structured_parts.append(f"Detail: {bp2.group(1).strip()}")
    if value and unit: structured_parts.append(f"Quantity: {value.group(1).strip()} {unit.group(1).strip()}")
    elif value: structured_parts.append(f"Value: {value.group(1).strip()}")
    
    cleaned_text = ". ".join(structured_parts)
    
    # Basic cleaning
    cleaned_text = cleaned_text.lower()
    cleaned_text = re.sub(r'[^\w\s.,:]', ' ', cleaned_text) # Keep important punctuation
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text) # Remove multiple spaces

    return cleaned_text.strip()

# --- PyTorch Dataset Class ---
class T5PriceDataset(Dataset):
    """
    PyTorch Dataset for T5 model.
    """
    def __init__(self, dataframe, tokenizer, source_max_len, target_max_len, is_test=False):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source_text = str(self.data.iloc[index]['t5_input'])
        
        source = self.tokenizer.batch_encode_plus(
            [source_text], max_length=self.source_max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()

        if self.is_test:
            return {'source_ids': source_ids.to(dtype=torch.long), 'source_mask': source_mask.to(dtype=torch.long)}

        target_text = str(self.data.iloc[index]['t5_target'])
        target = self.tokenizer.batch_encode_plus(
            [target_text], max_length=self.target_max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        target_ids = target['input_ids'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long)
        }

# --- PyTorch Lightning Model Definition ---
class T5PricePredictor(pl.LightningModule):
    """
    PyTorch Lightning module for the T5 model.
    """
    def __init__(self, model_name, learning_rate, tokenizer, train_dataset_len, batch_size, max_epochs):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.train_dataset_len = train_dataset_len
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits

    def training_step(self, batch, batch_idx):
        loss, _ = self(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_mask'],
            labels=batch['target_ids']
        )
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_mask'],
            labels=batch['target_ids']
        )
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        
        # Calculate total training steps for the scheduler
        num_gpus = self.trainer.num_devices if self.trainer else 1
        effective_batch_size = self.batch_size * num_gpus
        num_training_steps = (self.train_dataset_len // effective_batch_size) * self.max_epochs
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0, # Can be adjusted, e.g., int(num_training_steps * 0.1)
            num_training_steps=num_training_steps
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

# --- Main Execution Block ---
if __name__ == '__main__':
    # 1. Load Data
    try:
        train_df = pd.read_csv('dataset/train.csv', encoding='latin1')
        test_df = pd.read_csv('dataset/test.csv', encoding='latin1')
        print("Datasets loaded successfully.")
    except FileNotFoundError:
        print("Error: train.csv or test.csv not found in the 'dataset' directory.")
        exit()

    # 2. Preprocess and Format
    print("Using raw 'catalog_content' without cleaning as requested...")
    # Ensure content is string type to prevent errors with NaN values
    train_df['catalog_content'] = train_df['catalog_content'].astype(str)
    test_df['catalog_content'] = test_df['catalog_content'].astype(str)

    train_df['t5_input'] = "predict price: " + train_df['catalog_content']
    train_df['t5_target'] = train_df['price'].astype(str)
    test_df['t5_input'] = "predict price: " + test_df['catalog_content']
    
    # 3. Split Data
    train_split_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)
    print(f"Training set size: {len(train_split_df)}, Validation set size: {len(val_df)}")

    # 4. Initialize Tokenizer and Datasets
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    train_dataset = T5PriceDataset(train_split_df, tokenizer, SOURCE_MAX_LEN, TARGET_MAX_LEN)
    val_dataset = T5PriceDataset(val_df, tokenizer, SOURCE_MAX_LEN, TARGET_MAX_LEN)

    # 5. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    
    # 6. Initialize Model & Trainer
    model = T5PricePredictor(
        model_name=MODEL_NAME, 
        learning_rate=LEARNING_RATE, 
        tokenizer=tokenizer,
        train_dataset_len=len(train_dataset),
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best-model',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='bf16-mixed'  # Use bfloat16 mixed precision for H100
    )

    # 7. Train the Model
    print("\n--- Starting Model Training ---")
    trainer.fit(model, train_loader, val_loader)
    print("--- Training Finished ---")

    # 8. Inference on Test Set
    print("\n--- Generating Predictions on Test Set ---")
    best_model_path = checkpoint_callback.best_model_path
    trained_model = T5PricePredictor.load_from_checkpoint(best_model_path, tokenizer=tokenizer)
    trained_model.freeze()
    trained_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = T5PriceDataset(test_df, tokenizer, SOURCE_MAX_LEN, TARGET_MAX_LEN, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=8)
    
    predictions = []
    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids = batch['source_ids'].to(trained_model.device)
        attention_mask = batch['source_mask'].to(trained_model.device)
        
        generated_ids = trained_model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=TARGET_MAX_LEN,
            num_beams=5,
            early_stopping=True
        )
        
        preds = [
            tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        predictions.extend(preds)

    # 9. Create Submission File
    # Convert predictions to float, handling potential errors
    def to_float(price_str):
        try:
            return float(price_str)
        except (ValueError, TypeError):
            return 0.0 # Default to 0.0 if conversion fails

    test_df['price'] = [to_float(p) for p in predictions]
    # Ensure prices are positive
    test_df['price'] = test_df['price'].abs()

    submission_df = test_df[['sample_id', 'price']]
    submission_df.to_csv('submission.csv', index=False)

    print("\nSubmission file 'submission.csv' created successfully!")
    print(submission_df.head())

