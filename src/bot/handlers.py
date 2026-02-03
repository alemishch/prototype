import os
import httpx
from telegram import Update
from telegram.ext import ContextTypes
from src.core.utils import load_config

# We use an environment variable for the API URL so Docker can override it later
# Default to localhost for local testing
API_BASE = os.getenv("API_HOST", "http://127.0.0.1:8000")
API_URL = f"{API_BASE}/generate"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_text(f"Hello {user.first_name}! I am your LLM Abliteration research bot. Send me a prompt.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    
    # Send a placeholder message so the user knows something is happening
    status_msg = await update.message.reply_text("🧠 Thinking...")
    
    try:
        # Communicate with the FastAPI Node
        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "prompt": user_text,
                "max_new_tokens": 150,
                "temperature": 0.7
            }
            
            response = await client.post(API_URL, json=payload)
            response.raise_for_status()
            
            data = response.json()
            generated_text = data.get("text", "")
            
            if not generated_text.strip():
                generated_text = "[No text generated]"

            await status_msg.edit_text(generated_text)
            
    except httpx.ConnectError:
        await status_msg.edit_text("❌ Error: Could not connect to LLM API (Is the server running?)")
    except Exception as e:
        await status_msg.edit_text(f"❌ Error: {str(e)}")