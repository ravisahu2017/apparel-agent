#!/usr/bin/env python
# coding: utf-8

import os
import base64
import uuid
import sys
import requests
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
        tinydb_path=f"vision_data_{u}.nogit.json",
    )

    description = "generate a front pose of the modal in the mentioned kurti"
    if feedback:
        description += (
            f"\n\nCRITICAL FIXES REQUIRED BASED ON PREVIOUS ATTEMPT:\n{feedback}"
        )

    result = generator_chain.invoke(
        inputs={
            "description": description,
            "view": "front view",
            "market_place": "Meesho",
        },
        reference_images=[
            open(f"{INPUT_FOLDER}/front.png", "rb"),
            open(f"{INPUT_FOLDER}/neck.png", "rb"),
        ],
        output_path=f"output/{product_id}/generated_kurti_{u}_{attempt}.png",
    )
    return result


def extract():
    vision_chain = VisionExtractorChain(
        openrouter_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model="anthropic/claude-3-haiku",
        tinydb_path=f"vision_data_{uuid4}.nogit.json",
    )
    result = vision_chain.invoke(
        {"input_folder": INPUT_FOLDER, "product_id": product_id}
    )
    return result


def compare(u, attempt=1):
    comparer = CompareByVisionLLM(
        openrouter_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model="nvidia/nemotron-nano-12b-v2-vl:free",
    )
    raw_res = comparer.compare(
        f"{INPUT_FOLDER}/front.png",
        f"output/{product_id}/generated_kurti_{u}_{attempt}.png",
    )
    print(f"--- Comparer Raw Output ---\n{raw_res}")

    try:
        # Strip markdown code blocks if present
        clean_json = raw_res.strip()
        if clean_json.startswith("```"):
            clean_json = clean_json.split("\n", 1)[1].rsplit("```", 1)[0]

        return json.loads(clean_json)
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        return {
            "is_passed": False,
            "feedback_for_regeneration": "The output was not valid JSON. Please ensure the design details match exactly.",
        }


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
            current_feedback = comparison.get(
                "feedback_for_regeneration", "General mismatch in design."
            )
            print(f"\n❌ FAILED: Attempt {attempt} did not pass.")
            print(f"Issues: {comparison.get('issues')}")
            print(f"Feedback for next run: {current_feedback}")

            if attempt == max_attempts:
                print("\nReached maximum retry attempts. Manual intervention required.")


if __name__ == "__main__":
    step = "extract"
    if len(sys.argv) > 1:
        step = sys.argv[1]

    if step == "extract":
        extract()
    elif step == "generate":
        if len(sys.argv) < 3:
            print("Usage: python run_agent.py generate <uuid>")
            sys.exit(1)
        run_generation_loop(sys.argv[2])
    elif step == "compare":
        if len(sys.argv) < 3:
            print("Usage: python run_agent.py compare <uuid>")
            sys.exit(1)
        compare(sys.argv[2])
