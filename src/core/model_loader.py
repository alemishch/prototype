#### filepath: src/core/model_loader.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.core.utils import load_config

def get_device_and_config():
    """Determines the best configuration based on available hardware."""
    config = load_config()
    
    # Check if we actually have CUDA available
    if config['model']['device'] == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available. Swapping to CPU mode for local development.")
        return "cpu", False
        
    return config['model']['device'], config['model']['load_in_8bit']

def load_model_and_tokenizer():
    config = load_config()
    model_name = config['model']['name']
    hf_token = os.getenv("HF_TOKEN")
    
    device, use_8bit = get_device_and_config()
    
    print(f"Loading {model_name} on {device} (8-bit: {use_8bit})...")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model Loading Logic
    model_kwargs = {
        "token": hf_token,
        "trust_remote_code": True
    }

    if device == "cuda":
        model_kwargs["device_map"] = "auto"
        if use_8bit:
            model_kwargs["load_in_8bit"] = True
        else:
            model_kwargs["torch_dtype"] = torch.float16
    else:
        # Local Development Mock/CPU Fallback
        # If your local pc cannot handle 8B model, we might want to mock it.
        # For now, we attempt CPU load (slow, but works for checking code)
        model_kwargs["device_map"] = "cpu"
        model_kwargs["torch_dtype"] = torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        # Ensure model is in eval mode
        model.eval() 
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise e

    return model, tokenizer