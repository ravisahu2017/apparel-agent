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


class GradioOrchestrator:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.vision_chain = None
        self.generator_chain = None
        self.comparer = CompareByVisionLLM(os.getenv("OPENROUTER_API_KEY"))
        self.generated_images = []
        self.demo = self.create_gradio_interface()

    def create_gradio_interface(self):
        # --- GRADIO LAYOUT ---
        with gr.Blocks() as demo:
            gr.Markdown("# 👗 AI Apparel Agent")

            with gr.Row():
                status_box = gr.Textbox(label="Agent Status", interactive=False)
            with gr.Row():
                with gr.Column():
                    input_imgs = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="Upload Cleaned Kurti Images",
                    )
                    extract_btn = gr.Button("Extract", variant="primary")
                with gr.Column():
                    output_details = gr.Textbox(label="Extracted Attributes")

            with gr.Row(visible=False) as prompt_action_row:
                with gr.Column():
                    view_input = gr.Dropdown(
                        choices=["front view", "back view", "side view", "close-up view"],
                        value="front view",
                        label="Select View Type",
                        allow_custom_value=True
                    )
                    regenerate_prompt_btn = gr.Button(
                        "Regenerate Prompt for View", variant="secondary", size="sm"
                    )
                    generate_btn = gr.Button(
                        "Generate Image", variant="primary", size="md"
                    )
                    verify_btn = gr.Button(
                        "Verify & Refine", variant="secondary", size="md"
                    )

                with gr.Column():
                    image_gen_prompt = gr.Textbox(
                        label="Image Generation Prompt", visible=False
                    )
                    

            with gr.Row(visible=False) as generation_row:
                output_gallery = gr.Gallery(label="Generated Results", columns=3, rows=1, height="auto")

            # Connect the button to the function
            extract_btn.click(
                fn=self.process_extraction,
                inputs=[input_imgs],
                outputs=[
                    status_box,
                    output_details,
                    image_gen_prompt,
                    prompt_action_row,
                    generation_row,
                ],
            )

            generate_btn.click(
                fn=self.process_generation,
                inputs=[image_gen_prompt, input_imgs],
                outputs=[status_box, output_gallery],
            )

            verify_btn.click(
                fn=self.process_verification,
                inputs=[input_imgs, output_gallery, view_input],
                outputs=[status_box, output_details, image_gen_prompt],
            )

            regenerate_prompt_btn.click(
                fn=self.process_regenerate_prompt,
                inputs=[view_input],
                outputs=[status_box, output_details, image_gen_prompt],
            )
        return demo

    def process_extraction(self, input_images):
        if not input_images:
            return (
                "Please upload at least one image.",
                None,
                None,
                gr.update(visible=False),
                gr.update(visible=False),
            )

        # input_images is a list of file paths when select_compute is "files"
        if isinstance(input_images, str):
            input_images = [input_images]

        # 1. SETUP
        u = str(uuid.uuid4())
        p_id = "gradio_upload_" + u[:8]

        # STEP 1: UI Feedback
        yield gr.update(
            value="Step 1: Extracting Attributes from multiple images...", visible=True
        ), None, None, gr.update(visible=False), gr.update(visible=False)

        # 2. RUN EXTRACTION
        self.vision_chain = VisionExtractorChain(
            openrouter_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_model="anthropic/claude-3-haiku",
            tinydb_path=f"vision_data_{u}.nogit.json",
        )

        try:
            # Pass image_paths (plural) to trigger multi-image logic
            result = self.vision_chain.invoke(
                {"image_paths": input_images, "product_id": p_id}
            )

            extracted_attributes = result.get("raw", {})
            formatted_attributes = json.dumps(extracted_attributes, indent=2)

            # STEP 2: UI Feedback (Show extracted attributes)
            yield gr.update(
                value=f"Step 2: Attributes Extracted from {len(input_images)} images!"
            ), formatted_attributes, None, gr.update(visible=False), gr.update(
                visible=False
            )

        except Exception as e:
            yield gr.update(
                value=f"❌ Error during extraction: {str(e)}"
            ), None, None, gr.update(visible=False), gr.update(visible=False)
            return

        # STEP 3: UI Feedback (Generating)
        yield gr.update(
            value="Step 3: Generating Fashion Prompt..."
        ), formatted_attributes, None, gr.update(visible=False), gr.update(
            visible=False
        )

        # 3. RUN GENERATOR
        try:
            self.generator_chain = GeneratorChain(
                os.getenv("OPENAI_API_KEY"),
                os.getenv("OPENROUTER_API_KEY"),
                tinydb_path=f"vision_data_{u}.nogit.json",
            )

            # Use a temporary output path for the Gradio result
            output_filename = f"output/gradio_{u}.png"
            os.makedirs("output", exist_ok=True)

            # We use the first image as the primary reference for generation
            # or we could pass multiple if the generator supports it.
            # Based on previous logic, we use at least one reference.
            ref_files = [
                open(img, "rb") for img in input_images[:2]
            ]  # Take up to 2 for reference

            gen_result = self.generator_chain.generate_prompt(
                inputs={
                    "description": "generate a front pose of the modal in the mentioned kurti",
                    "view": "front view",
                    "market_place": "Meesho",
                }
            )
            cleaned_prompt = gen_result.get("cleaned_prompt", "no prompt generated")
            # Close file handles
            for f in ref_files:
                f.close()

            # FINAL OUTPUT
            yield gr.update(value="✅ Done!"), formatted_attributes, gr.update(
                value=cleaned_prompt, visible=True
            ), gr.update(visible=True), gr.update(visible=True)

        except Exception as e:
            yield gr.update(
                value=f"❌ Error during generation: {str(e)}"
            ), formatted_attributes, None, gr.update(visible=False), gr.update(
                visible=False
            )

    def process_generation(self, prompt, input_images):
        if not self.generator_chain:
            yield gr.update(value="❌ Please run extraction first"), self.generated_images
            return

        if not prompt or not input_images:
            yield gr.update(value="❌ Prompt and Images are required"), self.generated_images
            return

        yield gr.update(value="🎨 Generating image..."), gr.skip()

        try:
            output_filename = f"output/{self.session_id}/generated_{str(uuid.uuid4())}.png"
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)

            # Open reference images
            ref_files = [open(img, "rb") for img in input_images[:2]]

            try:
                self.generator_chain.generate_image(
                    prompt=prompt,
                    reference_images=ref_files,
                    output_path=output_filename,
                )
                self.generated_images.append(output_filename)
                yield gr.update(value="✅ Image Generated!"), self.generated_images
            finally:
                for f in ref_files:
                    f.close()

        except Exception as e:
            yield gr.update(value=f"❌ Error during generation: {str(e)}"), self.generated_images

    def process_regenerate_prompt(self, view):
        if not self.generator_chain:
            yield gr.update(value="❌ Please run extraction first"), None, None
            return

        yield gr.update(value=f"🔄 Regenerating {view} prompt..."), gr.skip(), gr.skip()

        try:
            gen_result = self.generator_chain.generate_prompt(
                inputs={
                    "description": f"generate a {view} of the modal in the mentioned kurti",
                    "view": view,
                    "market_place": "Meesho",
                }
            )
            cleaned_prompt = gen_result.get("cleaned_prompt", "no prompt generated")

            # Get current formatted attributes from the chain if possible, or just skip update
            yield gr.update(value="✅ Prompt Regenerated!"), gr.skip(), cleaned_prompt
        except Exception as e:
            yield gr.update(value=f"❌ Error: {str(e)}"), gr.skip(), gr.skip()

    def process_verification(self, input_images, gallery_data, view):
        if not self.generator_chain or not gallery_data:
            yield gr.update(value="❌ Run extraction and generation first"), gr.skip(), gr.skip()
            return

        # Use the most recent image from the gallery for verification
        generated_image_path = gallery_data[-1] if isinstance(gallery_data, list) else gallery_data
        # If it's a dict (Gradio gallery output), handle it
        if isinstance(generated_image_path, dict):
            generated_image_path = generated_image_path['name']

        yield gr.update(value="🔍 Verifying design consistency..."), gr.skip(), gr.skip()

        try:
            # We use the first image as the original reference for comparison
            original_image_path = input_images[0]
            
            raw_response = self.comparer.compare(original_image_path, generated_image_path)
            
            # Try to parse JSON from the response
            try:
                import re
                json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
                if json_match:
                    verification_data = json.loads(json_match.group())
                else:
                    verification_data = {"is_passed": False, "feedback_for_regeneration": raw_response}
            except:
                verification_data = {"is_passed": False, "feedback_for_regeneration": raw_response}

            is_passed = verification_data.get("is_passed", False)
            feedback = verification_data.get("feedback_for_regeneration", "No specific feedback")
            
            if is_passed:
                yield gr.update(value="✅ Verification Passed! Design matches."), gr.skip(), gr.skip()
            else:
                yield gr.update(value="⚠️ Verification Failed. Refining prompt..."), gr.skip(), gr.skip()
                
                # Use the feedback to regenerate the prompt
                gen_result = self.generator_chain.generate_prompt(
                    inputs={
                        "description": f"generate a {view} of the modal in the mentioned kurti",
                        "view": view,
                        "market_place": "Meesho",
                        "feedback": feedback
                    }
                )
                cleaned_prompt = gen_result.get("cleaned_prompt", "no prompt generated")
                
                yield gr.update(value="✅ Prompt Refined based on feedback! Click Generate again."), feedback, cleaned_prompt

        except Exception as e:
            yield gr.update(value=f"❌ Error during verification: {str(e)}"), gr.skip(), gr.skip()

    def run(self):
        self.demo.launch(
            server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft()
        )
