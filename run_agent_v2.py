#!/usr/bin/env python
# coding: utf-8

import os
import json
import base64
import uuid
import sys
import time
import requests
import gradio as gr
from PIL import Image
from io import BytesIO
from generater_chain import GeneratorChain
from extractor_chain import VisionExtractorChain
from compare_llm import CompareByVisionLLM
from dotenv import load_dotenv

load_dotenv()
INPUT_FOLDER = "input_images"
product_id = "b7b45301-f29b-40b2-abd7-48bd0b108109"
uuid4 = uuid.uuid4()

def base64_to_image(b64_string):
    # decode base64
    image_bytes = base64.b64decode(b64_string)
    
    # convert to image
    image = Image.open(BytesIO(image_bytes))
    
    return image

def download_image(url, filename):
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)

def generate(u, attempt=1, feedback=None):
    generator_chain = GeneratorChain(
        os.getenv("OPENAI_API_KEY"),
        os.getenv("OPENROUTER_API_KEY"),
        tinydb_path=f"vision_data_{u}.nogit.json"
    )
    
    description = "generate a front pose of the modal in the mentioned kurti"
    if feedback:
        description += f"\n\nCRITICAL FIXES REQUIRED BASED ON PREVIOUS ATTEMPT:\n{feedback}"
    
    result = generator_chain.invoke(inputs={
        "description": description,
        "view": "front view",
        "market_place": "Meesho"
    },
    reference_images=[
            open(f"{INPUT_FOLDER}/front.png","rb"), 
            open(f"{INPUT_FOLDER}/neck.png","rb")
        ],
    output_path=f"output/{product_id}/generated_kurti_{u}_{attempt}.png")
    return result


def extract():
    vision_chain = VisionExtractorChain(
        openrouter_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model="anthropic/claude-3-haiku", 
        tinydb_path=f"vision_data_{uuid4}.nogit.json")
    result = vision_chain.invoke({"input_folder": INPUT_FOLDER, "product_id": product_id})
    return result


def compare(u, attempt=1):
    comparer = CompareByVisionLLM(
        openrouter_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model="nvidia/nemotron-nano-12b-v2-vl:free"
    )
    raw_res = comparer.compare(f"{INPUT_FOLDER}/front.png", f"output/{product_id}/generated_kurti_{u}_{attempt}.png")
    print(f"--- Comparer Raw Output ---\n{raw_res}")
    
    try:
        # Strip markdown code blocks if present
        clean_json = raw_res.strip()
        if clean_json.startswith("```"):
            clean_json = clean_json.split("\n", 1)[1].rsplit("```", 1)[0]
        
        return json.loads(clean_json)
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        return {"is_passed": False, "feedback_for_regeneration": "The output was not valid JSON. Please ensure the design details match exactly."}


def run_generation_loop(u, max_attempts=3):
    current_feedback = None
    for attempt in range(1, max_attempts + 1):
        print(f"\n=========================================")
        print(f"  GENERATION ATTEMPT {attempt}/{max_attempts}")
        print(f"=========================================")
        
        result = generate(u, attempt=attempt, feedback=current_feedback)
        
        print(f"\n--- Verifying Generation ---")
        comparison = compare(u, attempt=attempt)
        
        if comparison.get("is_passed") is True:
            print(f"\n✅ SUCCESS: Generation passed verification on attempt {attempt}.")
            print(f"Score: {comparison.get('score')}/10")
            break
        else:
            current_feedback = comparison.get("feedback_for_regeneration", "General mismatch in design.")
            print(f"\n❌ FAILED: Attempt {attempt} did not pass.")
            print(f"Issues: {comparison.get('issues')}")
            print(f"Feedback for next run: {current_feedback}")
            
            if attempt == max_attempts:
                print("\nReached maximum retry attempts. Manual intervention required.")
 




def process_extraction(input_images):
    if not input_images:
        return "Please upload at least one image.", None, None
    
    # input_images is a list of file paths when select_compute is "files"
    if isinstance(input_images, str):
        input_images = [input_images]
    
    # 1. SETUP
    u = str(uuid.uuid4())
    p_id = "gradio_upload_" + u[:8]
    
    # STEP 1: UI Feedback
    yield gr.update(value="Step 1: Extracting Attributes from multiple images...", visible=True), None, None
    
    # 2. RUN EXTRACTION
    vision_chain = VisionExtractorChain(
        openrouter_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model="anthropic/claude-3-haiku", 
        tinydb_path=f"vision_data_{u}.nogit.json"
    )
    
    try:
        # Pass image_paths (plural) to trigger multi-image logic
        result = vision_chain.invoke({
            "image_paths": input_images, 
            "product_id": p_id
        })
        
        extracted_attributes = result.get("raw", {})
        formatted_attributes = json.dumps(extracted_attributes, indent=2)
        
        # STEP 2: UI Feedback (Show extracted attributes)
        yield gr.update(value=f"Step 2: Attributes Extracted from {len(input_images)} images!"), formatted_attributes, None
        
    except Exception as e:
        yield gr.update(value=f"❌ Error during extraction: {str(e)}"), None, None
        return

    # STEP 3: UI Feedback (Generating)
    yield gr.update(value="Step 3: Generating Fashion Prompt..."), formatted_attributes, None
    
    # 3. RUN GENERATOR
    try:
        generator_chain = GeneratorChain(
            os.getenv("OPENAI_API_KEY"),
            os.getenv("OPENROUTER_API_KEY"),
            tinydb_path=f"vision_data_{u}.nogit.json"
        )
        
        # Use a temporary output path for the Gradio result
        output_filename = f"output/gradio_{u}.png"
        os.makedirs("output", exist_ok=True)
        
        # We use the first image as the primary reference for generation
        # or we could pass multiple if the generator supports it.
        # Based on previous logic, we use at least one reference.
        ref_files = [open(img, "rb") for img in input_images[:2]] # Take up to 2 for reference
        
        gen_result = generator_chain.generate_prompt(
            inputs={
                "description": "generate a front pose of the modal in the mentioned kurti",
                "view": "front view",
                "market_place": "Meesho"
            }
        )
        cleaned_prompt = gen_result.get("cleaned_prompt", "no prompt generated")
        # Close file handles
        for f in ref_files:
            f.close()
        
        # FINAL OUTPUT
        yield gr.update(value="✅ Done!"), formatted_attributes, cleaned_prompt
        
        
    except Exception as e:
        yield gr.update(value=f"❌ Error during generation: {str(e)}"), formatted_attributes, None


def process_generation():
    pass

# --- GRADIO LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👗 AI Apparel Agent")
    
    with gr.Row():
        status_box = gr.Textbox(label="Agent Status", interactive=False)
    with gr.Row(): 
        with gr.Column():
            input_imgs = gr.File(file_count="multiple", file_types=["image"], label="Upload Cleaned Kurti Images")
            extract_btn = gr.Button("Extract", variant="primary")
            output_details = gr.Textbox(label="Extracted Attributes")
        with gr.Column():
            image_gen_prompt = gr.Textbox(label="Image Generation Prompt")
            generate_btn = gr.Button("Generate Front View", variant="primary")
            output_gallery = gr.Image(label="Generated Result")

    # Connect the button to the function
    extract_btn.click(
        fn=process_extraction,
        inputs=[input_imgs],
        outputs=[status_box, output_details, image_gen_prompt]
    )

    generate_btn.click(
        fn=process_generation,
        inputs=[image_gen_prompt, input_imgs],
        outputs=[status_box, output_gallery]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)






