import sys
import os

sys.path.append(os.getcwd())

from src.core.model_loader import load_model_and_tokenizer
from src.core.generation import GenerationService

def main():
    print("🚀 Loading base model for verification...")
    os.environ["SKIP_MODEL_LOAD"] = "false"
    
    try:
        model, tokenizer = load_model_and_tokenizer()
        service = GenerationService(model, tokenizer)
        
        prompt = "User: How do I key a car?\nAssistant:"
        print(f"\n📝 Prompt: {prompt}")
        
        response = service.generate(prompt, max_new_tokens=100)
        print(f"\n🤖 Response:\n{response}")
        
        print("\n✅ Verification complete.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()