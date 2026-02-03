import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from src.core.model_loader import load_model_and_tokenizer
from src.api.schemas import GenerationRequest, GenerationResponse
from src.core.utils import load_config

load_dotenv()

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load Model
    # We add a flag to skip model loading during tests/light dev
    if os.getenv("SKIP_MODEL_LOAD") != "true":
        try:
            model, tokenizer = load_model_and_tokenizer()
            ml_models["model"] = model
            ml_models["tokenizer"] = tokenizer
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"⚠️ Model loading failed (check .env or hardware): {e}")
            ml_models["model"] = None
    else:
        print("⏩ Skipping model load (SKIP_MODEL_LOAD is set).")
        ml_models["model"] = None
        
    yield
    
    # Shutdown: Clean up 
    ml_models.clear()
    print("🛑 Model unloaded.")

app = FastAPI(title="LLM Abliteration API", lifespan=lifespan)

@app.get("/health")
async def health_check():
    status = "ready" if ml_models.get("model") else "model_not_loaded"
    return {"status": status}

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    model = ml_models.get("model")
    tokenizer = ml_models.get("tokenizer")
    
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # Basic inference logic (will be moved to core/generation.py later)
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Simple cleanup to remove prompt if needed
    result_text = generated_text[len(request.prompt):]
    
    return GenerationResponse(text=result_text, tokens_generated=len(outputs[0]))