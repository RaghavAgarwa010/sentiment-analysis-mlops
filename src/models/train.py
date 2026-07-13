import argparse
import os

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import(
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)

MODEL_NAME="distillbert-base-uncased"
NUM_LABELS=2
MAX_LEN=256
PROCESSED_DIR="data/processed"
MODEL_SAVE_PATH="models/distillbert-sentiment"

class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":torch.tensor(self.labels[idx], dtype=torch.long),
        }
    
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels, total_loss=[], [], 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask= batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss/len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1, all_preds, all_labels

def train(args):
    device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
    logger.info(f"using device:{device}")

    logger.info("loading preprocessed data...")
    train_df = pd.read_csv(f"{PROCESSED_DIR}/train.csv")
    val_df = pd.read_csv(f"PROCESSED_DIR/val.csv")

    if args.dev_run:
        train_df= train_df.sample(500, random_state=42)
        val_df= val_df.sample(100, random_state=42)
        logger.warning("DEV RUN: using 500 train/ 100 val samples")

    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_dataset = SentimentDataset(train_df, tokenizer, MAX_LEN)
    val_dataset = SentimentDataset(train_df, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    logger.info(f"loading model: {MODEL_NAME}")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )