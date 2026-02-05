import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.core.utils import load_config

def load_model_and_tokenizer():
    config = load_config()
    model_name = config['model']['name']
    hf_token = os.getenv("HF_TOKEN")
    
    device = config['model']['device']
    use_8bit = config['model']['load_in_8bit']
    
    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available. Using CPU.")
        device = "cpu"
        use_8bit = False
    
    print(f"Loading {model_name} on {device} (8-bit: {use_8bit})...")

    # Load tokenizer (only AutoTokenizer, no deprecated LlamaTokenizer)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token=hf_token,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model configuration
    model_kwargs = {
        "token": hf_token,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True
    }

    if device == "cuda":
        model_kwargs["device_map"] = "auto"
        if use_8bit:
            model_kwargs["load_in_8bit"] = True
        else:
            # Use config value instead of hardcoding
            dtype_str = config['model']['torch_dtype']
            if dtype_str == "float16":
                model_kwargs["torch_dtype"] = torch.float16
            elif dtype_str == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16
            else:
                model_kwargs["torch_dtype"] = torch.float32
    else:
        model_kwargs["torch_dtype"] = torch.float32

    # Load model WITHOUT flash attention
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs) #, attn_implementation="flash_attention_2")
    model.eval()
    
    return model, tokenizer

