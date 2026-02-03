from pydantic import BaseModel
from typing import Optional, List

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    # abliteration control parameters
    # alpha: float = 1.0 

class GenerationResponse(BaseModel):
    text: str
    tokens_generated: int