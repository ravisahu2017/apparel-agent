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
Generate a high-quality, realistic product image of an Indian kurti (traditional long shirt).

PRODUCT SPECIFICATIONS:
=====================================
COLOR:
- Primary color: {primary_color}
- Accent color: {accent_color}

STYLE & FIT:
- Cut: {cut_style}
- Fit: {fit}
- Length: {length} (reaches hip level)
- Neckline: {neckline} with embroidered edge
- Sleeves: {sleeves} (plain with {motif} motif)

FABRIC:
- Type: {fabric_type}
- Finish: {finish}
- Texture: {texture}
- Weight: Light

PATTERN & DESIGN:
- Pattern type: {pattern_type} 
- Motif: {motif} woven pattern
- Density: {density} density
- Layout: {layout} across the garment
- Motif style: Tree-like, delicate ethnic designs

CONSTRUCTION DETAILS:
- Stitching: {stitching} stitching visible
- Hem: {border} border hem at bottom
- Overall quality: Premium, well-crafted

PRESENTATION:
- Show the complete kurti on a neutral white background
- Flat lay or hanging presentation to show full design
- Ensure all design elements (motifs, stitching, borders) are clearly visible
- Natural lighting, professional product photography style
- High resolution, sharp focus on fabric details

IMPORTANT:
- The kurti should look authentic and wearable
- All specified design elements must be clearly visible
- No model wearing the kurti (flat lay or standalone)
- Professional e-commerce product photography quality
"""
    
    return prompt.strip()

def get_filename(view):
    """Generate a unique filename for the generated image"""
    if view.find("front") != -1:
        view = "front_view"
    elif view.find("back") != -1:
        view = "back_view"
    elif view.find("side") != -1:
        view = "side_view"
    else: view = view.replace(" ", "_").lower()
    return f"{view}_{uuid.uuid4().hex[:8]}.png"

@tool("generate_image_from_design")
def generate_image_from_design(design_json, local_dir, output_dir, view="front view"):
    """
    Generate a realistic kurti image based on design specifications and multiple reference images.
    
    Args:
        design_json (dict): Design specifications containing color, style, fabric, pattern details
        reference_images (list): List of image file paths to use as input
        output_dir (str): Directory to save the generated image
    
    Returns:
        str: Path to the generated image
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found")
    
    # Build detailed prompt from design specifications
    prompt = build_generation_prompt(design_json)
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    log(f"Generating kurti image with prompt and multiple images...")

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

        filename = get_filename(view)
        output_path = os.path.join(output_dir, filename)

        with open(output_path, "wb") as f:
            f.write(image_bytes)
        log(f"✓ Kurti image generated successfully: {output_path}")
        return output_path
    except Exception as e:
        log(f"✗ Error generating image: {str(e)}", level="ERROR")
        raise

