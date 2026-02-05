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
        # Tokenize
        inputs = _tokenizer(user_text, return_tensors="pt")
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=_tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from response
        response = generated_text[len(user_text):].strip()
        
        if not response:
            response = "[No response generated]"
        
        await status_msg.edit_text(response[:4000])  # Telegram limit
        
    except Exception as e:
        await status_msg.edit_text(f"❌ Error: {str(e)}")