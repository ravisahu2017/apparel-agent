#!/usr/bin/env python
# coding: utf-8

import os
import base64
from PIL import Image
from io import BytesIO
from generater_chain import GeneratorChain
from extractor_chain import VisionExtractorChain
from dotenv import load_dotenv

from old_v2.agent_1 import product_id

load_dotenv()
INPUT_FOLDER = "input_images"
product_id = "a7a45301-f29b-40b2-abd7-48bd0b1081d9"
def base64_to_image(b64_string):
    # decode base64
    image_bytes = base64.b64decode(b64_string)
    
    # convert to image
    image = Image.open(BytesIO(image_bytes))
    
    return image


def generate():
    generator_chain = GeneratorChain(
        os.getenv("OPENAI_API_KEY"),
        os.getenv("OPENROUTER_API_KEY"),
        tinydb_path="vision_data.nogit.json"
    )
    # generator_chain.store_images_via_path([
    #     f"{INPUT_FOLDER}/back.png",
    #     f"{INPUT_FOLDER}/neck.png",
    #     f"{INPUT_FOLDER}/repeating_pattern.jpg",
    #     f"{INPUT_FOLDER}/front.png"
    # ]);
    
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
    image.save(f"output/{product_id}/generated_kurti.png")
    image.show()

def extract():
    vision_chain = VisionExtractorChain(
        openrouter_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model="anthropic/claude-3-haiku", 
        tinydb_path="vision_data.nogit.json")
    result = vision_chain.invoke({"input_folder": INPUT_FOLDER, "product_id": product_id})

    # fetch record from vision_chain's tinydb
    record = vision_chain.db.all()
    print("------------------------\nTinyDB records:\n", record[0])


    # fetch record from vision_chain's vectorstore
    vectorstore = vision_chain.vectorstore
    print("------------------------\nVectorstore records:\n", vectorstore.similarity_search("kurti"))

    return result


if __name__ == "__main__":
    print("------------------------\nExtracting attributes...\n")
    extract()
    print("------------------------\nAttributes:\n", attributes)
    print("-----------------------------------------\n")
    print("------------------------\nGenerating kurti...\n")
    #generate()






