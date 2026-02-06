import os
import torch
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from src.core.utils import load_config

def load_model_and_tokenizer():
    config_data = load_config()
    model_name = config_data['model']['name']
    hf_token = os.getenv("HF_TOKEN")
    
    device = config_data['model']['device']
    use_8bit = config_data['model']['load_in_8bit']
    
    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available. Using CPU.")
        device = "cpu"
        use_8bit = False
    
    print(f"Loading {model_name} on {device} (8-bit: {use_8bit})...")

    try:
        # 1. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=hf_token,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 2. explicit Config loading to fix potential artifacts
        # (Fixes the rope_scaling=null issue by letting AutoConfig handle it first)
        model_config = AutoConfig.from_pretrained(
            model_name, 
            token=hf_token, 
            trust_remote_code=True
        )

        model_kwargs = {
            "config": model_config, # Pass explicit config
            "token": hf_token,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            # Add offload folder to prevent "NoneType" errors in accelerate if VRAM is exceeded
            "offload_folder": "model_offload" 
        }

        if device == "cuda":
            # Use 'auto' to handle 16GB model on 12GB GPU (requires offloading)
            model_kwargs["device_map"] = "auto" 
            
            if use_8bit:
                model_kwargs["load_in_8bit"] = True
            else:
                dtype_str = config_data['model']['torch_dtype']
                if dtype_str == "float16":
                    model_kwargs["torch_dtype"] = torch.float16
                elif dtype_str == "bfloat16":
                    model_kwargs["torch_dtype"] = torch.bfloat16
                else:
                    model_kwargs["torch_dtype"] = torch.float32
        else:
            model_kwargs["torch_dtype"] = torch.float32

        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()
        
        return model, tokenizer

    except Exception as e:
        print(f"❌ Error detailed traceback:")
        traceback.print_exc()
        raise e