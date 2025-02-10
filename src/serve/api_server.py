from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
import uvicorn
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GPT-2 Chat API")

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

class ChatRequest(BaseModel):
    message: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40

class ChatResponse(BaseModel):
    response: str
    
@app.on_event("startup")
async def startup_event():
    global model, tokenizer, device
    try:
        # Set device
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load model and tokenizer
        model_path = "model_outputs/final_model"
        
        # Check if model path exists
        if not os.path.exists(model_path):
            raise Exception(f"Model path does not exist: {model_path}")
            
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Move model to device
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received message: {request.message}")
        
        # Encode the input text
        inputs = tokenizer(request.message, return_tensors="pt", padding=True, truncation=True)
        logger.info(f"Input shape: {inputs['input_ids'].shape}")
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            logger.info("Generating response...")
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=request.max_length + len(inputs["input_ids"][0]),
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                num_return_sequences=1,
                return_dict_in_generate=False  # Return tensor instead of object
            )
            logger.info(f"Output tensor shape: {outputs.size()}")
        
        # Get only the generated text (excluding the input prompt)
        input_length = len(inputs["input_ids"][0])
        generated_sequence = outputs[0][input_length:]
        
        # Decode the response
        response_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        logger.info(f"Generated response: {response_text}")
        
        return ChatResponse(response=response_text)
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/health")
async def health_check():
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "device": str(device)}

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True) 