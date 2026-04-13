#!/usr/bin/env python3
"""
FastAPI application for uploading images to S3 and integrating with existing MCP tools.
"""

import os
import uuid
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from typing import List
import tempfile

import mcp_client
from tools.s3_util import upload_file_object
from tinydb import TinyDB, Query
from tools.s3_util import list_s3_files, download_from_s3
from tools.tiny_db import update_record, insert_record
from tools.image_util import get_file_object

# Load environment variables
load_dotenv()

app = FastAPI(title="Apparel Agent API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def yield_output(streamingStatus, type, message = None, data = None):
    jsn = {"type": type, "streamingStatus": streamingStatus.lower()}
    if data:
        jsn['data'] = data
    if message:
        jsn['message'] = message
    return f"data: {json.dumps(jsn)}\n\n"

@app.post("/extract")
async def extract(files: List[UploadFile] = File(...)):
    async def event_generator():
        try:
            product_id = str(uuid.uuid4())
            temp_image_paths = []
            
            # --- PHASE 1: UPLOADING ---
            yield yield_output("uploading images to S3", "info")
            
            for file in files:
                spl = os.path.splitext(file.filename)
                file_extension = spl[1]
                unique_filename = f"{product_id}/{spl[0]}{file_extension}"

                # Save temp local copy for MCP tool
                temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension).name
                temp_image_paths.append(temp_file_path)
                content = await file.read() # Use await for UploadFile
                with open(temp_file_path, "wb") as f:
                    f.write(content)

                # Upload to S3
                # TODO: uncomment when S3 is ready - save billing for now
                upload_file_object(file.file, unique_filename, f"image/{file_extension}")

            # --- PHASE 2: DNA EXTRACTION ---
            yield yield_output("analyzing design DNA", "formatted_string", f"New product created\nproduct_id: {product_id}")
            
            dna = await mcp_client.extract_apparel_design(temp_image_paths, product_id)
            
            design_json = json.loads(dna)
            # Yield DNA immediately so UI can show it
            yield yield_output("design DNA extracted","json", "Design DNA extracted", design_json)
            
            insert_record({
                "product_id": product_id,
                "design_json": design_json,
            })
            
            # Assuming you have a prompt generation method in your mcp_client
            prompt_result = await mcp_client.generate_fashion_prompt(product_id, "front", design_json)
            
            # Save final state to TinyDB
            update_record(product_id, {
                "product_id": product_id,
                "prompt": prompt_result,
                "status": "prompt_generated"
            })

            # Final yield with full data
            yield yield_output("extraction complete", "formatted_string", "You can review this prompt and hit generate to generate the image", prompt_result)

        except Exception as e:
            yield yield_output("extraction failed", "formatted_string", "Extraction failed", str(e))

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/generate")
async def generate(product_id: str = Form(...), view: str = Form(...)):
    async def event_generator():
        print(f"Generating image for product: {product_id}")
        yield yield_output("generate started", "formatted_string", "Generating image")
        
        #fetch prompt from tiny db for product_id
        db = TinyDB("db/products.nogit.json")
        ProductQuery = Query()
        
        # Search for record with matching product_id
        records = db.search(ProductQuery.product_id == product_id)
        
        if not records:
            yield yield_output("generate failed", "formatted_string", f"No product found with ID: {product_id}")
            return
            
        product_record = records[0]
        prompt = product_record.get("prompt", "")


        # List all files in S3 with the product_id prefix
        s3_files = list_s3_files(product_id)
        
        if not s3_files or isinstance(s3_files, str):
            yield yield_output("generate failed", "formatted_string", f"No images found for product: {product_id}")
            return
            
        # Create temporary directory for downloaded images
        temp_dir = tempfile.mkdtemp(prefix=f"product_{product_id}_")
        input_images = []
        
        yield yield_output("downloading images", "formatted_string", f"Downloading {len(s3_files)} images...")
        
        # Download each file from S3
        for s3_key in s3_files:
            # Extract filename from S3 key
            filename = s3_key.split('/')[-1]
            local_path = os.path.join(temp_dir, filename)
            
            # Download file from S3
            download_result = download_from_s3(s3_key, local_path)
            
            if "Error" not in download_result and os.path.exists(local_path):
                input_images.append(local_path)
                print(f"Downloaded: {s3_key} -> {local_path}")
            else:
                print(f"Failed to download: {s3_key}")
        
        if not input_images:
            yield yield_output("generate failed", "formatted_string", "Failed to download any images")
            return
            
        yield yield_output("images ready", "formatted_string", f"Downloaded {len(input_images)} images for generation")

        result = await mcp_client.generate_fashion_image(prompt, input_images)
        
        image_file = get_file_object(result)
        s3_url = upload_file_object(image_file, f"{product_id}/generated/{view}_{uuid.uuid4()}.png", "image/png")
        data = {
            "product_id": product_id,
            "status": "image_generated"
        }
        if view.index("front") != -1:
            data[f"image_url"] = s3_url
        # Save final state to TinyDB
        update_record(product_id, data)
        yield yield_output("generate complete", "formatted_string", "Image generated successfully", {
            "image_url": s3_url
        })
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/fetch-products")
async def fetch_products():
    """
    Load products from TinyDB
    """
    try:
        db = TinyDB(f"db/products.nogit.json")
        products = db.all()
        return products
    except Exception as e:
        return []

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "service": "apparel-agent-api"}

if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
