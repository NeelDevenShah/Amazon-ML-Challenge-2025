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

MODEL_NAME = 't5-3b'
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_EPOCHS = 5
SOURCE_MAX_LEN = 256
TARGET_MAX_LEN = 8

# --- SMAPE Metric and Helper Functions ---
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """Calculate SMAPE - The competition metric."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    # Replace zeros in denominator with a small number to avoid division by zero
    denominator[denominator == 0] = 1e-8
    smape = np.mean(2 * np.abs(y_pred - y_true) / denominator) * 100
    return smape

def to_float(price_str):
    """Helper function to convert model output string to float."""
    try:
        # Handle cases where the model might output commas
        return float(str(price_str).replace(',', ''))
    except (ValueError, TypeError):
        return 0.0  # Default to 0.0 if conversion fails

# --- PyTorch Dataset Class ---
class T5PriceDataset(Dataset):
    """PyTorch Dataset for T5 model."""
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
    """PyTorch Lightning module for the T5 model with SMAPE validation."""
    def __init__(self, model_name, learning_rate, tokenizer, train_dataset_len, batch_size, max_epochs):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.train_dataset_len = train_dataset_len
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        # Store validation step outputs
        self.validation_step_outputs = []
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

        # Generate predictions to calculate SMAPE
        generated_ids = self.model.generate(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_mask'],
            max_length=TARGET_MAX_LEN,
            num_beams=3,  # Use a smaller beam size for faster validation
            early_stopping=True
        )
        
        preds = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        targets = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in batch['target_ids']]
        
        pred_prices = [to_float(p) for p in preds]
        target_prices = [to_float(t) for t in targets]

        output = {'preds': pred_prices, 'targets': target_prices}
        self.validation_step_outputs.append(output)
        return loss

    def on_validation_epoch_end(self):
        # Aggregate predictions and targets from all validation batches
        all_preds = []
        all_targets = []
        for output in self.validation_step_outputs:
            all_preds.extend(output['preds'])
            all_targets.extend(output['targets'])
        
        # Calculate SMAPE over the entire validation set
        val_smape = symmetric_mean_absolute_percentage_error(all_targets, all_preds)
        self.log('val_smape', val_smape, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()  # free memory
        
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        num_gpus = self.trainer.num_devices if self.trainer else 1
        effective_batch_size = self.batch_size * num_gpus
        num_training_steps = (self.train_dataset_len // effective_batch_size) * self.max_epochs
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

# --- Main Execution Block ---
# 1. Load Data
try:
    train_df = pd.read_csv('/root/train.csv', encoding='latin1')
    test_df = pd.read_csv('/root/test.csv', encoding='latin1')
    print("Datasets loaded successfully.")
except FileNotFoundError:
    print("Error: train.csv or test.csv not found in the 'dataset' directory.")
    exit()

# 2. Preprocess and Format
train_df['catalog_content'] = train_df['catalog_content'].astype(str)
test_df['catalog_content'] = test_df['catalog_content'].astype(str)
train_df['t5_input'] = "predict price: " + train_df['catalog_content']
train_df['t5_target'] = train_df['price'].astype(str)
test_df['t5_input'] = "predict price: " + test_df['catalog_content']

# 3. Split Data
train_split_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)

# 4. Initialize Tokenizer and Datasets
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
train_dataset = T5PriceDataset(train_split_df, tokenizer, SOURCE_MAX_LEN, TARGET_MAX_LEN)
val_dataset = T5PriceDataset(val_df, tokenizer, SOURCE_MAX_LEN, TARGET_MAX_LEN)

# 5. Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

# 6. Initialize Model & Trainer
model = T5PricePredictor(
    model_name=MODEL_NAME, learning_rate=LEARNING_RATE, tokenizer=tokenizer,
    train_dataset_len=len(train_dataset), batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS
)

checkpoint_callback = ModelCheckpoint(
    dirpath='/mnt/t5-main-2-large-model/checkpoints', filename='best-model-smape', save_top_k=1,
    verbose=True, monitor='val_smape', mode='min'
)

early_stopping_callback = EarlyStopping(monitor='val_smape', patience=10, mode='min')

trainer = pl.Trainer(
    callbacks=[checkpoint_callback, early_stopping_callback],
    max_epochs=MAX_EPOCHS, accelerator='gpu', devices=1, precision='bf16-mixed'
)

# 7. Train the Model
trainer.fit(model, train_loader, val_loader)

# 8. Inference on Test Set
best_model_path = checkpoint_callback.best_model_path
trained_model = T5PricePredictor.load_from_checkpoint(best_model_path, tokenizer=tokenizer)
trained_model.freeze()
trained_model.to('cuda' if torch.cuda.is_available() else 'cpu')

test_dataset = T5PriceDataset(test_df, tokenizer, SOURCE_MAX_LEN, TARGET_MAX_LEN, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=8)

predictions = []
for batch in tqdm(test_loader, desc="Predicting"):
    generated_ids = trained_model.model.generate(
        input_ids=batch['source_ids'].to(trained_model.device),
        attention_mask=batch['source_mask'].to(trained_model.device),
        max_length=TARGET_MAX_LEN, num_beams=5, early_stopping=True
    )
    preds = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
    predictions.extend(preds)

# 9. Create Submission File
test_df['price'] = [to_float(p) for p in predictions]
test_df['price'] = test_df['price'].abs()
submission_df = test_df[['sample_id', 'price']]
submission_df.to_csv('/mnt/t5-main-2-large-model/submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print(submission_df.head())

