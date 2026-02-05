import os
import logging
from dotenv import load_dotenv
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from src.bot.handlers import start, handle_message, set_model_and_tokenizer
from src.core.model_loader import load_model_and_tokenizer

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def main():
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not found!")
        return

    print("🔄 Loading model...")
    try:
        model, tokenizer = load_model_and_tokenizer()
        set_model_and_tokenizer(model, tokenizer)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    print("🤖 Starting bot...")
    app = ApplicationBuilder().token(token).build()

    import asyncio
    try:
        asyncio.run(app.bot.delete_webhook())
        print("ℹ️ Deleted existing webhook (if any).")
    except Exception as e:
        print(f"⚠️ Could not delete webhook: {e}")

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("✅ Bot is running!")
    app.run_polling()

if __name__ == "__main__":
    main()