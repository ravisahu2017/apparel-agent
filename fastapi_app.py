#!/usr/bin/env python3
"""
FastAPI application for uploading images to S3 and integrating with existing MCP tools.
"""

import os
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from typing import List
import tempfile

import mcp_client
from tools.s3_util import upload_file_object
from tinydb import TinyDB, Query


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

@app.post("/extract")
async def extract(files: List[UploadFile] = File(...)):
    """
    Upload multiple images to S3 and process them with existing MCP tools.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        JSON response with processing results or error message
    """
    try:
        # Upload files to S3
        print(f"Uploading {len(files)} files to S3...")
        upload_results = []
        s3_urls = []
        temp_image_paths = []
        product_id = str(uuid.uuid4())
        for file in files:
            # Generate unique filename
            spl = os.path.splitext(file.filename)
            file_extension = spl[1]
            file_name = spl[0]
            unique_filename = f"{product_id}/{file_name}{file_extension}"

            temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension).name
            temp_image_paths.append(temp_file_path)
            with open(temp_file_path, "wb") as f:
                f.write(file.file.read())

            # Upload to S3
            s3_url = upload_file_object(file.file, unique_filename, f"image/{file_extension}")
            s3_urls.append(s3_url)
            upload_results.append({
                "filename": file_name,
                "s3_url": s3_url
            })
        
        # Call MCP extraction tool
        extraction_result = await mcp_client.extract_apparel_design(temp_image_paths, product_id)
        
        # Save product data to TinyDB
        db = TinyDB(f"db/products.nogit.json")
        product_data = {
            "product_id": product_id,
            "timestamp": str(uuid.uuid4()),
            "extraction_result": extraction_result,
            "status": "extracted"
        }

        db.insert(product_data)
        print(f"Saved product data to TinyDB: {product_id}")

        return JSONResponse(content={**product_data,
            "message": "Images uploaded and processed successfully",
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process images: {str(e)}"}
        )


@app.post("/fetch-products")
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
