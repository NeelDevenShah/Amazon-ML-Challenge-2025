import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    T5EncoderModel,
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

# --- SMAPE Metric ---
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """Calculate SMAPE - The competition metric."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    denominator[denominator == 0] = 1e-8
    smape = np.mean(2 * np.abs(y_pred - y_true) / denominator) * 100
    return smape

# --- Enhanced text cleaning function ---
def clean_text_enhanced(text):
    """Cleans and structures the catalog content to extract key information."""
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

# --- PyTorch Dataset Class ---
class T5EncoderDataset(Dataset):
    """Dataset for T5 Encoder with log-transformed targets."""
    def __init__(self, dataframe, tokenizer, source_max_len, is_test=False):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_max_len = source_max_len
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

        if not self.is_test:
            price = float(self.data.iloc[index]['price'])
            log_price = np.log1p(price)  # log(1 + price) transformation
            result['log_price'] = torch.tensor(log_price, dtype=torch.float32)
            result['price'] = torch.tensor(price, dtype=torch.float32)  # Keep original for SMAPE calculation
        
        return result

# --- MLP Regression Head ---
class MLPRegressionHead(nn.Module):
    """Simple MLP for regression from encoder embeddings."""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))  # Output: log(price)
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mlp(x).squeeze(-1)

# --- PyTorch Lightning Model ---
class T5EncoderMLPLogPredictor(pl.LightningModule):
    """T5 Encoder + MLP with log-transformed target."""
    def __init__(self, model_name, learning_rate, tokenizer, train_dataset_len, batch_size, max_epochs):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        
        # Load T5 encoder only (not the full seq2seq model)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.tokenizer = tokenizer
        
        # Freeze 70% of encoder layers for efficiency
        total_params = list(self.encoder.parameters())
        freeze_count = int(len(total_params) * 0.7)
        for param in total_params[:freeze_count]:
            param.requires_grad = False
        
        # MLP regression head
        encoder_dim = self.encoder.config.d_model
        self.regression_head = MLPRegressionHead(
            encoder_dim, 
            hidden_dims=[512, 256, 128], 
            dropout=0.3
        )
        
        # Loss function - Huber loss is robust to outliers
        self.loss_fn = nn.HuberLoss(delta=1.0)
        
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask):
        # Get encoder outputs
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state
        
        # Mean pooling over sequence dimension
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * attention_mask_expanded, dim=1)
        sum_mask = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled_output = sum_hidden / sum_mask
        
        # Predict log(price)
        log_price_pred = self.regression_head(pooled_output)
        
        return log_price_pred

    def training_step(self, batch, batch_idx):
        log_price_pred = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(log_price_pred, batch['log_price'])
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        log_price_pred = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(log_price_pred, batch['log_price'])
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        
        # Convert log predictions back to original scale for SMAPE
        price_pred = torch.expm1(log_price_pred)  # expm1(x) = exp(x) - 1, inverse of log1p
        
        self.validation_step_outputs.append({
            'preds': price_pred.cpu().numpy(),
            'targets': batch['price'].cpu().numpy()
        })
        
        return loss

    def on_validation_epoch_end(self):
        all_preds = np.concatenate([out['preds'] for out in self.validation_step_outputs])
        all_targets = np.concatenate([out['targets'] for out in self.validation_step_outputs])
        
        # Clip predictions to be non-negative
        all_preds = np.clip(all_preds, 0, None)
        
        val_smape = symmetric_mean_absolute_percentage_error(all_targets, all_preds)
        self.log('val_smape', val_smape, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        
        num_training_steps = (self.hparams.train_dataset_len // self.hparams.batch_size) * self.hparams.max_epochs
        num_warmup_steps = int(num_training_steps * 0.05)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def predict_step(self, batch, batch_idx):
        log_price_pred = self(batch['input_ids'], batch['attention_mask'])
        price_pred = torch.expm1(log_price_pred)
        return price_pred

# --- Main Execution ---
if __name__ == '__main__':
    print("=" * 80)
    print("APPROACH 1: Flan-T5 Encoder + MLP + Log-Transform")
    print("=" * 80)
    
    # 1. Load Data
    train_df = pd.read_csv('/root/train.csv', encoding='latin1')
    test_df = pd.read_csv('/root/test.csv', encoding='latin1')

    # 2. Preprocess
    print("\n📝 Applying enhanced text cleaning...")
    train_df['cleaned_content'] = train_df['catalog_content'].astype(str).apply(clean_text_enhanced)
    test_df['cleaned_content'] = test_df['catalog_content'].astype(str).apply(clean_text_enhanced)
    train_df['t5_input'] = "predict price: " + train_df['cleaned_content']
    test_df['t5_input'] = "predict price: " + test_df['cleaned_content']

    # 3. Split Data
    train_split_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)
    print(f"📊 Training: {len(train_split_df)}, Validation: {len(val_df)}")

    # 4. Initialize
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    train_dataset = T5EncoderDataset(train_split_df, tokenizer, SOURCE_MAX_LEN)
    val_dataset = T5EncoderDataset(val_df, tokenizer, SOURCE_MAX_LEN)

    # 5. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 6. Model & Trainer
    model = T5EncoderMLPLogPredictor(
        model_name=MODEL_NAME, learning_rate=LEARNING_RATE, tokenizer=tokenizer,
        train_dataset_len=len(train_dataset), batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='/mnt/approach1-encoder-mlp-log/checkpoints', 
        filename='best-encoder-mlp-log', 
        save_top_k=1, verbose=True, monitor='val_smape', mode='min'
    )
    early_stopping_callback = EarlyStopping(monitor='val_smape', patience=4, mode='min')

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=MAX_EPOCHS, accelerator='gpu', devices=1, precision='bf16-mixed'
    )

    # 7. Train
    print("\n🚀 Training Flan-T5 Encoder + MLP with Log-Transform...")
    trainer.fit(model, train_loader, val_loader)

    # 8. Inference
    print("\n🔮 Starting inference on test set...")
    best_model = T5EncoderMLPLogPredictor.load_from_checkpoint(
        checkpoint_callback.best_model_path, tokenizer=tokenizer
    )
    best_model.freeze()
    best_model.eval()
    best_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = T5EncoderDataset(test_df, tokenizer, SOURCE_MAX_LEN, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=8)

    predictions = []
    for batch in tqdm(test_loader, desc="Predicting"):
        batch = {k: v.to(best_model.device) for k, v in batch.items()}
        with torch.no_grad():
            pred = best_model.predict_step(batch, 0)
            predictions.extend(pred.cpu().numpy())

    # 9. Submission
    test_df['price'] = np.array(predictions).clip(min=0)
    submission_df = test_df[['sample_id', 'price']]
    submission_df.to_csv('/mnt/approach1-encoder-mlp-log/submission_approach1.csv', index=False)

    print("\n✅ Approach 1 Complete! Submission saved.")
    print(submission_df.head())
    print(f"\n📈 Price Statistics:")
    print(f"   Min: ${submission_df['price'].min():.2f}")
    print(f"   Max: ${submission_df['price'].max():.2f}")
    print(f"   Mean: ${submission_df['price'].mean():.2f}")
    print(f"   Median: ${submission_df['price'].median():.2f}")