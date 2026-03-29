import os
import base64
from PIL import Image
from io import BytesIO
from generater_chain import GeneratorChain
from extractor_chain import VisionExtractorChain
from dotenv import load_dotenv
    
import json
import base64
from pathlib import Path
from tinydb import TinyDB
from langchain.schema.runnable import RunnableLambda
from langchain_community.vectorstores import Chroma
from clip_embeddings import CLIPEmbeddings



def get_image_files(input_folder):
    # Get all images from folder
    image_files = []
    if os.path.exists(input_folder):
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        for file in os.listdir(input_folder):
            if Path(file).suffix.lower() in image_extensions:
                full_path = os.path.join(input_folder, file)
                image_files.append(full_path)
                print("DEBUG", f"Found image: {file} ({os.path.getsize(full_path)} bytes)")
    else:
        return {"error": f"Folder not found: {input_folder}"}
    
    if not image_files:
        return {"error": f"No images found in {input_folder}. Checked extensions: jpg, jpeg, png, gif, webp"}
    
    print("INFO", f"Total images found: {len(image_files)}")
    return image_files

def add_images_to_vectorstore(input_folder):
        texts = []
        metadatas = []
        ids = []
        s3_prefix = f"https://rs-apparels.s3.ap-south-1.amazonaws.com/cleaned/"

        image_files = get_image_files(input_folder)

        for i, path in enumerate(image_files):
            view = path.split(".")[0]

            texts.append(f"{view} view of kurti")   # 👈 REQUIRED for RAG
            metadatas.append({
                "type": path,
                "view": view,
                "product_id": "asdasd",
                "s3_url": s3_prefix + path
            })
            ids.append(f"img_{i}")
        print("adding images to vector", ids, texts, metadatas)


        vectorstore = Chroma(
            collection_name="abc1",
            embedding_function=CLIPEmbeddings()
        )
        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        return vectorstore

    
if __name__ == "__main__":
    add_images_to_vectorstore("input_images")
    #read from vector store
    vectorstore = Chroma(
        collection_name="abc1",
        embedding_function=CLIPEmbeddings()
    )
    print("----------\nVectorstore records:")
    print(vectorstore.similarity_search("kurti"))