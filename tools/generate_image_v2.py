import json
import os
import uuid
import base64
from langchain.tools import tool
from openai import OpenAI
from tools.utils.log import log
from prompts.get_prompt import get_prompt

def build_generation_prompt(design_json):
    """
    Build a detailed, comprehensive prompt for image generation based on design specs.
    
    Args:
        design_json (dict): Design specifications
    
    Returns:
        str: Detailed prompt for image generation
    """
    
    # Extract design elements
    color = design_json.get("color", {})
    style = design_json.get("style", {})
    fabric = design_json.get("fabric", {})
    pattern = design_json.get("pattern", {})
    details = design_json.get("details", {})
    
    primary_color = color.get("primary", "pink")
    accent_color = color.get("accents", "white")
    cut_style = style.get("cut", "straight")
    fit = style.get("fit", "regular")
    length = style.get("length", "hip-length")
    neckline = style.get("neckline", "V-neck")
    sleeves = style.get("sleeves", "3/4th")
    
    fabric_type = fabric.get("type", "cotton")
    finish = fabric.get("finish", "matte")
    texture = fabric.get("texture", "smooth")
    
    pattern_type = pattern.get("type", "ethnic")
    motif = pattern.get("motif", "tree-like")
    density = pattern.get("density", "medium")
    layout = pattern.get("layout", "all-over")
    
    stitching = details.get("stitching", "visible")
    border = details.get("border", "plain")
    
    prompt = f"""
    {get_prompt("lock_design_contract")}
    {get_prompt("universal_negative")}
    
    Use this design JSON to generate a model image: 
    {json.dumps(design_json, indent=2)}
"""
    
    return prompt.strip()

@tool("generate_image_from_design")
def generate_image_from_design(design_json, local_dir, output_dir, output_filename, user_prompt):
    """
    Generate a realistic kurti image based on design specifications and multiple reference images.
    
    Args:
        design_json (dict): Design specifications containing color, style, fabric, pattern details
        local_dir (str): Path to directory containing reference images
        output_dir (str): Directory to save the generated image
        output_filename (str): Name of the output image file
        user_prompt (str): Additional user instructions for image generation
    
    Returns:
        str: Path to the generated image
    """
    generate_image(
        design_json=design_json,
        local_dir=local_dir,
        output_dir=output_dir,
        output_filename=output_filename,
        user_prompt=user_prompt
    )

def generate_image(design_json, local_dir, output_dir, output_filename, user_prompt):
    #validate design_json
    if not isinstance(design_json, dict):
        log("Design JSON must be a dictionary", level="ERROR")
        raise ValueError("Design JSON must be a dictionary")

    #print all arguments
    log(f"Design JSON: {json.dumps(design_json, indent=2)}")
    log(f"Local dir: {local_dir}")
    log(f"Output dir: {output_dir}")
    log(f"Output filename: {output_filename}")
    log(f"User prompt: {user_prompt}")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found")
    
    # Build detailed prompt from design specifications
    prompt = build_generation_prompt(design_json)
    prompt = f"{prompt}\n\n{user_prompt}"
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    log(f"Generating image with prompt and multiple images...")

    images = []
    #for each image in local_dir, read and convert to base64
    for img_path in os.listdir(local_dir):
        full_path = os.path.join(local_dir, img_path)
        if os.path.isfile(full_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            images.append(open(full_path, "rb"))
                
    log(f"{len(images)} reference images provided in {local_dir}")
    try:
        result = client.images.edit(
            model="gpt-image-1.5",
            image=images,  # Pass list of base64-encoded images
            prompt=prompt,
            size="1024x1024"
        )   
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, "wb") as f:
            f.write(image_bytes)
        log(f"✓ Image generated successfully: {output_path}")
        return output_path
    except Exception as e:
        log(f"✗ Error generating image: {str(e)}", level="ERROR")
        raise

