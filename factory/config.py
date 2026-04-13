import os
from dotenv import load_dotenv

load_dotenv()

# CENTRAL TRUTH FOR MODELS
MODELS = {
    "vision": [{
        "id": "nvidia/nemotron-nano-12b-v2-vl:free",
        "provider": "openrouter",
        "temperature": 0.1
    },{
        "id": "anthropic/claude-3-haiku",
        "provider": "openrouter",
        "temperature": 0.1
    },{
        "id": "gemini-2.5-flash",  # Current stable high-speed vision
        "provider": "google",
        "temperature": 0.1
    },
    {
        "id": "gemini-3-flash-preview", # Cutting edge high-speed
        "provider": "google",
        "temperature": 0.1
    }],
    "text": [{
        "id": "gemini-2.5-flash",  # Current stable high-speed vision
        "provider": "google",
        "temperature": 0.1
    },
    {
        "id": "gemini-3-flash-preview", # Cutting edge high-speed
        "provider": "google",
        "temperature": 0.1
    }],
    "image_edit": [{
        "id": "black-forest-labs/FLUX.2-flex", # or flux.2-flex
        "provider": "siliconflow",
        "temperature": 0.1
    },{
        "id": "black-forest-labs/FLUX.1-schnell", # or flux.2-flex
        "provider": "siliconflow",
        "temperature": 0.1
    },{
        "id": "black-forest-labs/FLUX.2-pro", # or flux.2-flex
        "provider": "siliconflow",
        "temperature": 0.1
    },{
        "id": "black-forest-labs/FLUX.1-dev", # or flux.2-flex
        "provider": "siliconflow",
        "temperature": 0.1
    }]
}