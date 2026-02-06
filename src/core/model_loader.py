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

        # 2. explicit Config loading & Patching
        model_config = AutoConfig.from_pretrained(
            model_name, 
            token=hf_token, 
            trust_remote_code=True
        )

        # PATCH: Fix for Llama-3 "AttributeError: 'NoneType' object has no attribute 'get'"
        # Accelerate fails if rope_scaling is None in some versions during device mapping
        if not hasattr(model_config, "rope_scaling") or model_config.rope_scaling is None:
            # We don't change the value, but ensuring the attribute exists safely for checks
            pass 

        model_kwargs = {
            "config": model_config,
            "token": hf_token,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True, 
            "offload_folder": "/app/model_offload" # Absolute path to be safe
        }

        if device == "cuda":
            model_kwargs["device_map"] = "auto"
            
            # Adding load_in_8bit=False explicitly can sometimes confuse older bnb versions
            if use_8bit:
                model_kwargs["load_in_8bit"] = True
            else:
                dtype_str = config_data['model'].get('torch_dtype', 'float16')
                if dtype_str == "bfloat16":
                    model_kwargs["torch_dtype"] = torch.bfloat16
                else:
                    model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["device_map"] = "cpu"
            model_kwargs["torch_dtype"] = torch.float32

        # 3. Load Model
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()
        
        return model, tokenizer

    except Exception as e:
        print(f"❌ Error detailed traceback:")
        traceback.print_exc()
        raise e