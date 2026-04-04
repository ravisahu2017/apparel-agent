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
    }],
    "generation_prompt": [{
        "id": "anthropic/claude-3-haiku",
        "provider": "openrouter",
        "temperature": 0.5
    }],
    "image_edit": [{
        "id": "black-forest-labs/FLUX.2-flex", # or flux.2-flex
        "provider": "siliconflow",
        "temperature": 0.1
    }]
}

# API KEYS
API_KEYS = {
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
    "siliconflow": os.getenv("SILICONFLOW_API_KEY"),
}