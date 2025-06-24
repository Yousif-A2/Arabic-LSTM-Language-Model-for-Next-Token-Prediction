"""
Input validation and configuration validation utilities
"""

import re
from typing import Dict, Any, Tuple

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate application configuration"""
    required_keys = ['app', 'model', 'interface']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate model config
    model_config = config['model']
    if 'max_length' not in model_config or model_config['max_length'] <= 0:
        raise ValueError("Invalid max_length in model config")
    
    if 'temperature_range' not in model_config:
        raise ValueError("Missing temperature_range in model config")
    
    temp_range = model_config['temperature_range']
    if len(temp_range) != 2 or temp_range[0] >= temp_range[1]:
        raise ValueError("Invalid temperature_range format")
    
    return True

def validate_arabic_text(text: str) -> Tuple[bool, str]:
    """Validate Arabic text input"""
    if not text or not text.strip():
        return False, "Text cannot be empty"
    
    # Check for Arabic characters
    arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
    arabic_chars = len(re.findall(arabic_pattern, text))
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars == 0:
        return False, "No alphabetic characters found"
    
    if arabic_chars / total_chars < 0.5:
        return False, "Text should be primarily in Arabic"
    
    # Check length
    words = text.split()
    if len(words) > 50:
        return False, "Text too long (max 50 words)"
    
    return True, "Valid Arabic text"

def sanitize_input(text: str) -> str:
    """Sanitize user input"""
    # Remove potentially harmful characters
    text = re.sub(r'[<>"\']', '', text)
    
    # Limit length
    text = text[:1000]
    
    return text.strip()