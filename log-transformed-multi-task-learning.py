import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
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
pl.seed_everything(42)

MODEL_NAME = 'google/flan-t5-xl'
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
MAX_EPOCHS = 50
SOURCE_MAX_LEN = 256
TARGET_MAX_LEN = 8

# Price buckets for classification
PRICE_BUCKETS = [0, 10, 20, 50, 100, 500, 10000]
NUM_BUCKETS = len(PRICE_BUCKETS) - 1

# --- SMAPE Metric ---
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """Calculate SMAPE."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    denominator[denominator == 0] = 1e-8
    smape = np.mean(2 * np.abs(y_pred - y_true) / denominator) * 100
    return smape

def to_float(price_str):
    """Convert string to float."""
    try:
        return float(str(price_str).replace(',', ''))
    except (ValueError, TypeError):
        return 0.0

def price_to_bucket(price):
    """Map price to bucket index."""
    for i, upper_bound in enumerate(PRICE_BUCKETS[1:]):
        if price < upper_bound:
            return i
    return NUM_BUCKETS - 1

# --- Enhanced text cleaning ---
def clean_text_enhanced(text):
    """Extract key information from catalog content."""
    if pd.isnull(text):
        return ""
    
    item_name = re.search(r"Item Name:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    bp1 = re.search(r"Bullet Point\s*1:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    bp2 = re.search(r"Bullet Point\s*2:\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
    prod_desc = re.search(r"Product Description:\s*(.*?)(?=\nValue:|\nUnit:|$)", text, re.DOTALL | re.IGNORECASE)
    value = re.search(r"Value:\s*([\d.,]+)", text, re.IGNORECASE)
    unit = re.search(r"Unit:\s*([A-Za-z]+)", text, re.IGNORECASE)
    
    structured_parts = []
    if item_name: structured_parts.append(f"Item: {item_name.group(1).strip()}")
    if bp1: structured_parts.append(f"Feature: {bp1.group(1).strip()}")
    if bp2: structured_parts.append(f"Detail: {bp2.group(1).strip()}")
    if prod_desc: structured_parts.append(f"Description: {prod_desc.group(1).strip()[:300]}")
    if value and unit: structured_parts.append(f"Quantity: {value.group(1).strip()} {unit.group(1).strip()}")
    elif value: structured_parts.append(f"Value: {value.group(1).strip()}")
    
    cleaned_text = ". ".join(structured_parts)
    cleaned_text = cleaned_text.lower()
    cleaned_text = re.sub(r'[^\w\s.,:]', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text.strip()

# --- Dataset ---
class T5MultiTaskDataset(Dataset):
    """Dataset with multi-task labels."""
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
        
        result = {
            'input_ids': source['input_ids'].squeeze().to(dtype=torch.long),
            'attention_mask': source['attention_mask'].squeeze().to(dtype=torch.long)
        }

        if self.is_test:
            return result

        # For training: add both exact price target and bucket classification
        price = float(self.data.iloc[index]['price'])
        log_price = np.log1p(price)
        bucket_idx = price_to_bucket(price)
        
        # Target for text generation (log-transformed price)
        target_text = f"{log_price:.4f}"
        target = self.tokenizer.batch_encode_plus(
            [target_text], max_length=self.target_max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        
        result.update({
            'labels': target['input_ids'].squeeze().to(dtype=torch.long),
            'price': torch.tensor(price, dtype=torch.float32),
            'log_price': torch.tensor(log_price, dtype=torch.float32),
            'bucket_idx': torch.tensor(bucket_idx, dtype=torch.long)
        })
        
        return result

# --- Multi-Task Model ---
class T5MultiTaskPredictor(pl.LightningModule):
    """T5 with multi-task learning: exact log-price generation + bucket classification."""
    def __init__(self, model_name, learning_rate, tokenizer, train_dataset_len, batch_size, max_epochs):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = tokenizer
        
        # Add classification head on top of encoder
        encoder_dim = self.model.config.d_model
        self.bucket_classifier = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_BUCKETS)
        )
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask, labels=None):
        # Get encoder hidden states
        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden = encoder_outputs.last_hidden_state
        
        # Main T5 generation loss (for log-transformed price)
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels,
            encoder_outputs=encoder_outputs
        )
        
        # Bucket classification from encoder
        # Use mean pooling
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(encoder_hidden.size()).float()
        sum_hidden = torch.sum(encoder_hidden * attention_mask_expanded, dim=1)
        sum_mask = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_hidden / sum_mask
        
        bucket_logits = self.bucket_classifier(pooled)
        
        return outputs.loss, outputs.logits, bucket_logits

    def training_step(self, batch, batch_idx):
        gen_loss, _, bucket_logits = self(
            batch['input_ids'], batch['attention_mask'], batch['labels']
        )
        
        # Auxiliary bucket classification loss
        bucket_loss = self.ce_loss(bucket_logits, batch['bucket_idx'])
        
        # Combined loss: 80% generation, 20% bucket classification
        total_loss = 0.8 * gen_loss + 0.2 * bucket_loss
        
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_gen_loss', gen_loss, on_epoch=True)
        self.log('train_bucket_loss', bucket_loss, on_epoch=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        gen_loss, _, bucket_logits = self(
            batch['input_ids'], batch['attention_mask'], batch['labels']
        )
        
        bucket_loss = self.ce_loss(bucket_logits, batch['bucket_idx'])
        total_loss = 0.8 * gen_loss + 0.2 * bucket_loss
        
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True)
        
        # Generate predictions
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=TARGET_MAX_LEN,
            num_beams=3,
            early_stopping=True
        )
        
        preds = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
        
        # Convert log predictions to actual prices
        log_prices_pred = [to_float(p) for p in preds]
        prices_pred = [np.expm1(lp) for lp in log_prices_pred]  # exp(log_price) - 1
        
        self.validation_step_outputs.append({
            'preds': np.array(prices_pred),
            'targets': batch['price'].cpu().numpy()
        })
        
        return total_loss

    def on_validation_epoch_end(self):
        all_preds = np.concatenate([out['preds'] for out in self.validation_step_outputs])
        all_targets = np.concatenate([out['targets'] for out in self.validation_step_outputs])
        
        all_preds = np.clip(all_preds, 0, None)
        
        val_smape = symmetric_mean_absolute_percentage_error(all_targets, all_preds)
        self.log('val_smape', val_smape, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        num_training_steps = (self.hparams.train_dataset_len // self.hparams.batch_size) * self.hparams.max_epochs
        num_warmup_steps = int(num_training_steps * 0.05)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

# --- Main Execution ---
if __name__ == '__main__':
    print("=" * 80)
    print("APPROACH 2: Log-Transform + Multi-Task Learning")
    print("=" * 80)
    
    # 1. Load Data
    train_df = pd.read_csv('/root/train.csv', encoding='latin1')
    test_df = pd.read_csv('/root/test.csv', encoding='latin1')

    # 2. Preprocess
    print("\n📝 Applying enhanced text cleaning...")
    train_df['cleaned_content'] = train_df['catalog_content'].astype(str).apply(clean_text_enhanced)
    test_df['cleaned_content'] = test_df['catalog_content'].astype(str).apply(clean_text_enhanced)
    train_df['t5_input'] = "predict log price: " + train_df['cleaned_content']
    test_df['t5_input'] = "predict log price: " + test_df['cleaned_content']

    # 3. Split
    train_split_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)
    print(f"📊 Training: {len(train_split_df)}, Validation: {len(val_df)}")

    # 4. Initialize
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    train_dataset = T5MultiTaskDataset(train_split_df, tokenizer, SOURCE_MAX_LEN, TARGET_MAX_LEN)
    val_dataset = T5MultiTaskDataset(val_df, tokenizer, SOURCE_MAX_LEN, TARGET_MAX_LEN)

    # 5. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 6. Model
    model = T5MultiTaskPredictor(
        model_name=MODEL_NAME, learning_rate=LEARNING_RATE, tokenizer=tokenizer,
        train_dataset_len=len(train_dataset), batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='/mnt/approach2-log-multitask/checkpoints',
        filename='best-log-multitask',
        save_top_k=1, verbose=True, monitor='val_smape', mode='min'
    )
    early_stopping_callback = EarlyStopping(monitor='val_smape', patience=4, mode='min')

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=MAX_EPOCHS, accelerator='gpu', devices=1, precision='bf16-mixed'
    )

    # 7. Train
    print("\n🚀 Training with Log-Transform + Multi-Task Learning...")
    trainer.fit(model, train_loader, val_loader)

    # 8. Inference
    print("\n🔮 Starting inference...")
    best_model = T5MultiTaskPredictor.load_from_checkpoint(
        checkpoint_callback.best_model_path, tokenizer=tokenizer
    )
    best_model.freeze()
    best_model.eval()
    best_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = T5MultiTaskDataset(test_df, tokenizer, SOURCE_MAX_LEN, TARGET_MAX_LEN, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=8)

    predictions = []
    for batch in tqdm(test_loader, desc="Predicting"):
        batch = {k: v.to(best_model.device) for k, v in batch.items()}
        with torch.no_grad():
            generated_ids = best_model.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=TARGET_MAX_LEN,
                num_beams=5,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            log_prices = [to_float(p) for p in preds]
            prices = [np.expm1(lp) for lp in log_prices]
            predictions.extend(prices)

    # 9. Submission
    test_df['price'] = np.array(predictions).clip(min=0)
    submission_df = test_df[['sample_id', 'price']]
    submission_df.to_csv('/mnt/approach2-log-multitask/submission_approach2.csv', index=False)

    print("\n✅ Approach 2 Complete! Submission saved.")
    print(submission_df.head())
    print(f"\n📈 Price Statistics:")
    print(f"   Min: ${submission_df['price'].min():.2f}")
    print(f"   Max: ${submission_df['price'].max():.2f}")
    print(f"   Mean: ${submission_df['price'].mean():.2f}")
    print(f"   Median: ${submission_df['price'].median():.2f}")