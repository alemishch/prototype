#!/bin/bash
set -e

SERVICE_NAME=$1

if [ "$SERVICE_NAME" = "api" ]; then
    echo "Starting FastAPI Server..."
    exec uvicorn src.api.server:app --host 0.0.0.0 --port 8000
    
elif [ "$SERVICE_NAME" = "bot" ]; then
    echo "Starting Telegram Bot..."
    exec python src/bot/telegram_bot.py
    
elif [ "$SERVICE_NAME" = "test" ]; then
    echo "Running Tests..."
    exec pytest src/tests/
    
else
    echo "Unknown service: $SERVICE_NAME"
    echo "Usage: docker run <image> [api|bot|test]"
    exit 1
fi