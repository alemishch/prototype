import torch
from telegram import Update
from telegram.ext import ContextTypes

# Global state for model/tokenizer
_model = None
_tokenizer = None

def set_model_and_tokenizer(model, tokenizer):
    global _model, _tokenizer
    _model = model
    _tokenizer = tokenizer

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_text(
        f"Hello {user.first_name}! 🤖\n"
        f"I'm running Llama locally. Send me a message!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _model is None or _tokenizer is None:
        await update.message.reply_text("❌ Model not loaded!")
        return
    
    user_text = update.message.text
    status_msg = await update.message.reply_text("🧠 Generating...")
    
    try:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        inputs = _tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
        
        
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.05,
                pad_token_id=_tokenizer.eos_token_id,
                eos_token_id=_tokenizer.eos_token_id
            )
        
        # FIX: Get only the new tokens
        input_length = inputs['input_ids'].shape[1]
        response_tokens = outputs[0][input_length:]
        response = _tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        if not response.strip():
            response = "[No response generated]"
        
        await status_msg.edit_text(response[:4000])
        
    except Exception as e:
        await status_msg.edit_text(f"❌ Error: {str(e)}")