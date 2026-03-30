#!/usr/bin/env python
# coding: utf-8

import os
import base64
import uuid
import sys
from PIL import Image
from io import BytesIO
from generater_chain import GeneratorChain
from extractor_chain import VisionExtractorChain
from verifier_llm import VisionLLMExtractor
from compare_llm import CompareByVisionLLM
from dotenv import load_dotenv

from old_v2.agent_1 import product_id

load_dotenv()
INPUT_FOLDER = "input_images"
product_id = "a7a45301-f29b-40b2-abd7-48bd0b1081d9"
uuid4 = uuid.uuid4()
def base64_to_image(b64_string):
    # decode base64
    image_bytes = base64.b64decode(b64_string)
    
    # convert to image
    image = Image.open(BytesIO(image_bytes))
    
    return image


def generate(u):
    generator_chain = GeneratorChain(
        os.getenv("OPENAI_API_KEY"),
        os.getenv("OPENROUTER_API_KEY"),
        tinydb_path=f"vision_data_{u}.nogit.json"
    )
    
    result = generator_chain.invoke({
        "description": "generate a front pose of the modal in the mentioned kurti",
        "view": "front view",
        "market_place": "Meesho"
    })
    
    print("hitting image generation api for below prompt\n")
    print("\n---------------result------------\n", result)

    
    response = generator_chain.generate_with_reference(
        result["cleaned_prompt"], 
        img=[
            open(f"{INPUT_FOLDER}/front.png","rb"), 
            open(f"{INPUT_FOLDER}/neck.png","rb")
        ]
    )
    image = base64_to_image(response)
    image.save(f"output/{product_id}/generated_kurti_{u}.png")
    image.show()

def extract():
    vision_chain = VisionExtractorChain(
        openrouter_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model="anthropic/claude-3-haiku", 
        tinydb_path=f"vision_data_{uuid4}.nogit.json")
    result = vision_chain.invoke({"input_folder": INPUT_FOLDER, "product_id": product_id})

    # fetch record from vision_chain's tinydb
    record = vision_chain.db.all()
    print("------------------------\nTinyDB records:\n", record[0])


    # fetch record from vision_chain's vectorstore
    vectorstore = vision_chain.vectorstore
    print("------------------------\nVectorstore records:\n", vectorstore.similarity_search("kurti"))

    return result


def verify():
    verifier = VisionLLMExtractor(
        openrouter_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model="anthropic/claude-3-haiku", 
    )
    result = verifier.extract(f"./output/generated_kurti.png")
    print("------------------------\nVerifier result:\n", result)


def compare(u):
    comparer = CompareByVisionLLM(
        openrouter_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model="anthropic/claude-3-haiku", 
    )
    result = comparer.compare(f"{INPUT_FOLDER}/front.png", f"output/{product_id}/generated_kurti_{u}.png")
    print("------------------------\nComparer result:\n", result)


if __name__ == "__main__":
    step = "extract"
    if len(sys.argv) > 1:
        step = sys.argv[1]
    
    if step == "extract":
        print("------------------------\nExtracting attributes...\n")
        extract()
        print("-----------------------------------------\n")
    elif step == "generate":
        print("------------------------\nGenerating kurti...\n")
        u = sys.argv[2]
        generate(u)
    elif step == "verify":
        print("------------------------\nVerifying kurti...\n")
        verify()
    elif step == "compare":
        print("------------------------\nComparing kurti...\n")
        u = sys.argv[2]
        compare(u)







