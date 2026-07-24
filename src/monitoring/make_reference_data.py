import os
import pandas as pd
from loguru import logger

PROCESSED_DIR = "data/processed"
REFERENCE_DIR = "data/reference"
SAMPLE_SIZE = 3000

def load_processed_data():
    logger.info("Loading processed train and test splits...")
    train_df = pd.read_csv(f"{PROCESSED_DIR}/train.csv")
    test_df = pd.read_csv(f"{PROCESSED_DIR}/test.csv")
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    logger.info(f"Combined dataset size: {len(combined_df)} rows")
    return combined_df

def stratified_sample(combined_df, sample_size):
    logger.info(f"Sampling {sample_size} rows, stratified by label...")
    per_class = sample_size // 2
    sampled_df = combined_df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(n=per_class, random_state=42)
    )
    sampled_df = sampled_df.reset_index(drop=True)
    logger.info(f"Class balance:\n{sampled_df['label'].value_counts()}")
    return sampled_df

def save_reference(sampled_df):
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    output_path = f"{REFERENCE_DIR}/reference_data.csv"
    sampled_df = sampled_df.rename(columns={"text": "review_text"})
    sampled_df.to_csv(output_path, index=False)
    logger.success(f"Reference dataset saved: {len(sampled_df)} rows to {output_path}")

def main():
    combined_df = load_processed_data()
    sampled_df = stratified_sample(combined_df, SAMPLE_SIZE)
    save_reference(sampled_df)

if __name__ == "__main__":
    main()
