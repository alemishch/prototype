import yaml
import os
from pathlib import Path

def load_config(config_path: str = None) -> dict:
    if config_path is None:
        root_dir = Path(__file__).parent.parent.parent
        config_path = root_dir / "src" / "config" / "settings.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config