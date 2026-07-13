import os
import pandas as pd
from datasets import load_dataset
from loguru import logger

RAW_DIR="data/raw"
PROCESSED_DIR="data/processed"

def download_datasets():
    logger.info("Downloading IMDb dataset from HuggingFace...")
    dataset=load_dataset("stanfordnlp/imdb")
    logger.info(f"dataset loaded: {dataset}")
    return dataset

def preprocess(dataset):
    logger.info("preprocessing dataset splits...")

    train_df=pd.DataFrame({
        "text": dataset["train"]["text"],
        "label": dataset["train"]["label"]
    })
    test_df=pd.DataFrame({
        "test": dataset["test"]["text"],
        "label": dataset["test"]["label"]
    })

    val_df= train_df.sample(frac=0.1, random_state=42)
    train_df = train_df.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    logger.info(f"train :{len(train_df)}, Val:{len(val_df)}, test: {len(test_df)}")
    return train_df, val_df, test_df

def save_splits(train_df, val_df, test_df):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_df.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
    val_df.to_csv(f"{PROCESSED_DIR}/val.csv", index=False)
    test_df.to_csv(f"{PROCESSED_DIR}/test.csv", index=False)

    logger.success(f"Saved splits to {PROCESSED_DIR}/")

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    dataset = download_datasets()
    train_df, val_df, test_df = preprocess(dataset)
    save_splits(train_df, val_df, test_df)
    logger.success("Dataset preparation complete!")

if __name__ == "__main__":
    main()

