import os
import json
import base64
from pathlib import Path
from langchain.tools import tool
from openai import OpenAI

@tool("extract_design_parameters", description="Extracts apparel design parameters from images using vision model and returns JSON")
def extract_design_parameters(image_folder_path):
    """
    Extracts apparel design parameters from all images in a folder using GPT-4o vision model.
    Returns structured JSON with design attributes.
    """
    # Load API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY environment variable not set"}
    
    print(f"[DEBUG] API Key loaded: {len(api_key)} characters")
    print(f"[DEBUG] Looking for images in: {image_folder_path}")
    
    client = OpenAI(api_key=api_key)
    
    # Get all images from folder
    image_files = []
    if os.path.exists(image_folder_path):
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        for file in os.listdir(image_folder_path):
            if Path(file).suffix.lower() in image_extensions:
                full_path = os.path.join(image_folder_path, file)
                image_files.append(full_path)
                print(f"[DEBUG] Found image: {file} ({os.path.getsize(full_path)} bytes)")
    else:
        return {"error": f"Folder not found: {image_folder_path}"}
    
    if not image_files:
        return {"error": f"No images found in {image_folder_path}. Checked extensions: jpg, jpeg, png, gif, webp"}
    
    print(f"[DEBUG] Total images found: {len(image_files)}")
    
    # Convert images to base64
    image_content = []
    for img_path in sorted(image_files):
        try:
            with open(img_path, "rb") as img_file:
                image_data = img_file.read()
                base64_image = base64.b64encode(image_data).decode("utf-8")
                
                # Determine image type from extension
                file_ext = Path(img_path).suffix.lower()
                if file_ext in ['.jpg', '.jpeg']:
                    media_type = "image/jpeg"
                elif file_ext == '.png':
                    media_type = "image/png"
                elif file_ext == '.gif':
                    media_type = "image/gif"
                elif file_ext == '.webp':
                    media_type = "image/webp"
                else:
                    media_type = "image/jpeg"
                
                image_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{base64_image}",
                    },
                })
                print(f"[DEBUG] Encoded image: {Path(img_path).name}")
        except Exception as e:
            print(f"[ERROR] Error reading image {img_path}: {e}")
            return {"error": f"Failed to read image {img_path}: {str(e)}"}
    
    # Add text prompt
    image_content.append({
        "type": "text",
        "text": """Analyze these apparel product images and extract design parameters in this exact JSON format:
{
  "product_type": "kurti/suit/top/dress",
  "style": {
    "cut": "straight/A-line/fitted/flared/oversized",
    "length": "short/knee-length/hip-length/long",
    "fit": "regular/slim/loose",
    "sleeve_type": "full/3-4th/half/sleeveless",
    "neck_style": "round/V/boat/square"
  },
  "fabric": {
    "type": "cotton/silk/wool/synthetic/blend",
    "texture": "smooth/plain/textured/embroidered",
    "weight": "light/medium/heavy",
    "finish": "matte/semi-matte/shiny"
  },
  "color": {
    "primary": "color name and tone",
    "secondary": ["color list if any"],
    "accents": ["accent colors if any"]
  },
  "pattern": {
    "type": "solid/floral/paisley/geometric/ethnic/abstract/print",
    "technique": "flat print/woven/knitted/embroidery",
    "density": "sparse/medium/dense",
    "layout": "all-over/border/panel/neck-only",
    "repeat_unit": "description of smallest pattern element"
  },
  "design_details": {
    "neckline": "description",
    "border_hem": "description",
    "sleeves_detail": "description",
    "stitching": "visible or not",
    "special_features": ["any special features"]
  },
  "confidence": "high/medium/low"
}

Analyze all visible details WITHOUT adding interpretations or suggestions. Be precise and factual."""
    })
    
    try:
        print("[DEBUG] Sending request to GPT-4o vision model...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": image_content,
                }
            ],
            max_tokens=2000,
        )
        
        print("[DEBUG] Received response from GPT-4o")
        
        # Extract JSON from response
        response_text = response.choices[0].message.content
        print(f"[DEBUG] Response text length: {len(response_text)}")
        
        # Try to parse JSON from response
        try:
            # Find JSON in response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                design_params = json.loads(json_str)
                print("[DEBUG] Successfully parsed JSON")
                return design_params
            else:
                return {
                    "raw_response": response_text,
                    "error": "Could not find JSON in response"
                }
        except json.JSONDecodeError as je:
            print(f"[ERROR] JSON parse error: {je}")
            return {
                "raw_response": response_text,
                "error": f"JSON parse error: {str(je)}"
            }
            
    except Exception as e:
        print(f"[ERROR] Vision model error: {type(e).__name__}: {e}")
        return {"error": f"Vision model error: {type(e).__name__}: {str(e)}"}
    
   