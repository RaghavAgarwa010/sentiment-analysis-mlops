import argparse
import torch
from loguru import logger
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

MODEL_SAVE_PATH = "models/distilbert-sentiment"
MAX_LEN = 256
LABELS= {0: "negative", 1: "positive"}

def load_model(model_path: str = MODEL_SAVE_PATH):
    logger.info(f"loading model from{model_path}")
    tokenizer= DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

def predict(text: str,model, tokenizer, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
    model.to(device)

    encoding=tokenizer(
        text, 
        max_length= MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids= input_ids, attention_mask= attention_mask)
        probs= torch.softmax(outputs.logits, dim=1).squeeze(0)
        pred= torch.argmax(probs).item()

    return {
        "label":      LABELS[pred],
        "confidence": round(probs[pred].item(), 4),
        "scores": {
            "negative": round(probs[0].item(), 4),
            "positive": round(probs[1].item(), 4),
        },
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sentiment inference on a text input")
    parser.add_argument("--text", type=str, required=True, help="Text to classify")
    parser.add_argument("--model_path", type=str, default= MODEL_SAVE_PATH)
    args=parser.parse_args()

    model, tokenizer= load_model(args.model_path)
    result=predict(args.text, model, tokenizer)
    logger.info(f"Input: {args.text}")
    logger.info(f"prediction: {result['label']} (confidence: {result['confidence']}) ")
    logger.info(f"Scores: {result['scores']}")

