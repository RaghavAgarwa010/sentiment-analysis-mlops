from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager
from loguru import logger
import sys
import os
from .logger_utils import log_prediction

sys.path.append(os.path.join(os.path.dirname(__file__),'..', 'src', 'models'))

from predict import predict, load_model
model_store={}
from pydantic import field_validator

class SentimentRequest(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def text_must_not_be_empty(cls, v):
        if v.strip() == '':
            raise ValueError('text must not be empty')
        return v

class SentimentResponse(BaseModel):
    label:str
    confidence:float
    scores:dict

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model()
    model_store["model"] = model
    model_store["tokenizer"] = tokenizer
    model_store["predictor"] = predict
    logger.info("Model ready.")
    yield

app=FastAPI(title="Sentiment Analysis API", lifespan=lifespan)

@app.post("/predict", response_model=SentimentResponse)
async def predict_endpoint(request: SentimentRequest, background_tasks: BackgroundTasks):
    try:
        predictor=model_store["predictor"]
        result = predictor(request.text, model_store["model"], model_store["tokenizer"])

        background_tasks.add_task(
            log_prediction,
            text=request.text,
            label=result['label'],
            confidence= result['confidence']
        )

        return SentimentResponse(
            label=result['label'],
            confidence=result["confidence"],
            scores=result["scores"]
        )
    except Exception as e:
        logger.error(f"prediction failed: {e}")
        raise HTTPException(status_code=500,detail=str(e))
    