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

MODEL_NAME="distilbert-base-uncased"
NUM_LABELS=2
MAX_LEN=256
PROCESSED_DIR="data/processed"
MODEL_SAVE_PATH="models/distilbert-sentiment"

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
    val_df = pd.read_csv(f"{PROCESSED_DIR}/val.csv")

    if args.dev_run:
        train_df= train_df.sample(500, random_state=42)
        val_df= val_df.sample(100, random_state=42)
        logger.warning("DEV RUN: using 500 train/ 100 val samples")

    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_dataset = SentimentDataset(train_df, tokenizer, MAX_LEN)
    val_dataset = SentimentDataset(val_df, tokenizer, MAX_LEN)

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
    best_val_f1= 0.0
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    with mlflow.start_run():
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "max_len": MAX_LEN,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "dev_run":args.dev_run,
        })

        for epoch in range(1, args.epochs+1):
            model.train()
            total_train_loss=0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs=model(input_ids = input_ids, attention_mask= attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()

            avg_train_loss= total_train_loss / len(train_loader)
            val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, val_loader, device)

            logger.info(
                f"Epoch {epoch}/ {args.epochs} | "
                f"train_loss: {avg_train_loss:.4f} | "
                f"val_loss: {val_loss: .4f} | "
                f"val_acc: {val_acc: .4f} | "
                f"val_f1: {val_f1: .4f}"
            )

            mlflow.log_metrics({
                "train_loss" : avg_train_loss,
                "val_loss" : val_loss,
                "val_acc" : val_acc,
                "val_f1" : val_f1
            }, step=epoch)

            if val_f1> best_val_f1:
                best_val_f1 = val_f1
                model.save_pretrained(MODEL_SAVE_PATH)
                tokenizer.save_pretrained(MODEL_SAVE_PATH)
                logger.info(f" new best  model saved (val_f1={best_val_f1: .4f})")

        logger.info("── Final classification report (val) ──")
        _, _, _, val_preds, val_labels= evaluate(model, val_loader, device)
        logger.info(f"\n {classification_report(val_labels, val_preds, target_names=['negative', 'positive'])}")
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE,
        )
        logger.info(f"MLFlow run complete. Best val_f1: {best_val_f1: .4f}")


if __name__== "__main__":
    parser= argparse.ArgumentParser(description="Train DistilBERT sentiment classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dev_run", action="store_true",
                        help= "Use small data subset for a quick smoke test")
    args = parser.parse_args()
    train(args)

