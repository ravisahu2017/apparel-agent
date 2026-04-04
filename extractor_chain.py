import os
import json
import base64
from pathlib import Path
from tinydb import TinyDB
from factory import ModelFactory
from langchain_core.runnables import RunnableLambda
from langchain_chroma import Chroma
from clip_embeddings import CLIPEmbeddings


class VisionExtractorChain:
    """
    Chain for:
    1. Reading image
    2. Extracting attributes via OpenRouter vision model
    3. Saving structured output into TinyDB
    4. Storing summary text in Chroma Vectorstore
    """

    def __init__(
        self,
        tinydb_path="vision_data.json",
        vectorstore_collection="images",
    ):
        self.vectorstore_collection = vectorstore_collection
        self.db = TinyDB(tinydb_path)
        self.embeddings = CLIPEmbeddings()

        self.vectorstore = Chroma(
            collection_name="vision_attributes", embedding_function=self.embeddings
        )

    def get_image_files(self, input_folder):
        # Get all images from folder
        image_files = []
        if os.path.exists(input_folder):
            image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
            for file in os.listdir(input_folder):
                if Path(file).suffix.lower() in image_extensions:
                    full_path = os.path.join(input_folder, file)
                    image_files.append(full_path)
        else:
            return {"error": f"Folder not found: {input_folder}"}

        if not image_files:
            return {
                "error": f"No images found in {input_folder}. Checked extensions: jpg, jpeg, png, gif, webp"
            }

        print("INFO", f"Total images found: {len(image_files)}")
        return image_files

    def parse_response(self, inputs):
        """
        Parse and clean the response from vision model to ensure valid JSON output

        Args:
            inputs: Dictionary containing the raw response from extract_attributes

        Returns:
            Dictionary with clean JSON attributes
        """
        raw_response = inputs.get("raw_response", "")

        if not raw_response:
            return {"error": "No response to parse"}

        # Try to parse as JSON directly
        try:
            attributes = json.loads(raw_response)
            if isinstance(attributes, dict):
                return attributes
        except json.JSONDecodeError:
            pass

        # Handle multiline JSON or text with JSON blocks
        cleaned_response = raw_response.strip()

        # Remove common prefixes/suffixes
        prefixes_to_remove = [
            "```json",
            "```",
            "Here's the JSON:",
            "JSON response:",
            "Response:",
            "Result:",
        ]

        for prefix in prefixes_to_remove:
            if cleaned_response.startswith(prefix):
                cleaned_response = cleaned_response[len(prefix) :].strip()

        # Find JSON blocks in text
        json_start = cleaned_response.find("{")
        json_end = cleaned_response.rfind("}") + 1

        if json_start != -1 and json_end > json_start:
            json_str = cleaned_response[json_start:json_end]
            try:
                attributes = json.loads(json_str)
                if isinstance(attributes, dict):
                    return attributes
            except json.JSONDecodeError:
                pass

        # Try to extract JSON from each line
        lines = cleaned_response.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    attributes = json.loads(line)
                    if isinstance(attributes, dict):
                        return attributes
                except json.JSONDecodeError:
                    continue

        # If all parsing fails, return as raw text
        return {
            "raw": cleaned_response,
            "error": "Could not parse JSON from response",
            "parsing_failed": True,
        }

    # ---------------------------------------------
    # STEP 1 → Load image file
    # ---------------------------------------------
    def load_image(self, inputs):
        # Handle multiple image paths (for Gradio UI), single path, or folder
        if "image_paths" in inputs:
            image_files = inputs["image_paths"]
        elif "image_path" in inputs:
            image_files = [inputs["image_path"]]
        else:
            input_folder = inputs["input_folder"]
            image_files = self.get_image_files(input_folder)
            image_files = sorted(image_files)

        # Convert images to base64
        image_content = []
        for img_path in image_files:
            try:
                with open(img_path, "rb") as img_file:
                    image_data = img_file.read()
                    base64_image = base64.b64encode(image_data).decode()

                    # Determine image type from extension
                    file_ext = Path(img_path).suffix.lower()
                    if file_ext in [".jpg", ".jpeg"]:
                        media_type = "image/jpeg"
                    elif file_ext == ".png":
                        media_type = "image/png"
                    elif file_ext == ".gif":
                        media_type = "image/gif"
                    elif file_ext == ".webp":
                        media_type = "image/webp"
                    else:
                        media_type = "image/jpeg"

                    filename = Path(img_path).name
                    print("INFO", f"Processing image: {filename}")
                    image_content.append(
                        {"type": "text", "text": f"Next image: {filename}"}
                    )
                    image_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_image}",
                                "name": filename,
                            },
                        }
                    )
                    print("INFO", f"Encoded image: {Path(img_path).name}")
            except Exception as e:
                print("ERROR", f"Error reading image {img_path}: {e}")
                return {"error": f"Failed to read image {img_path}: {str(e)}"}
        inputs["image_content"] = image_content
        return inputs

    # ---------------------------------------------
    # STEP 2 → Vision model (OpenRouter)
    # ---------------------------------------------
    def extract_attributes(self, inputs):
        """
        Extract attributes from images using OpenRouter vision model

        Args:
            inputs: Dictionary containing image_content list

        Returns:
            Dictionary with extracted attributes
        """
        image_content = inputs["image_content"]
        print("INFO", "Extracting attributes from images", len(image_content))

        prompt = """
        You are a fashion vision model. You have been given a different images of a kurti. Analyze the kurti item in the images
        and extract the following attributes:

        - Type of garment: kurti
        - Silhouette: A-line, fit, straight, etc.
        - Patterns  
        - Colors  
        - Sleeves(3/4th, Full, Sleeveless, Half)
        - Top length (Crop, Midi, short Midi, Long Midi, Maxi, etc.)
        - Neck design (Round, V-Neck, deep v, etc.)
        - Border hem details
        - Notable visual details  
        - Style category: casual, formal, ethnic, etc.  
        - Keywords  

        Return a JSON object with the extracted attributes.

        """

        content_array = [
            {
                "type": "text",
                "text": "Analyze this image and extract the requested attributes.",
            },
            *image_content,
        ]
        print("INFO", "Content array length:", len(content_array))

        response = ModelFactory.call_model("vision", prompt, content_array)
        if response:
            parsed_response = self.parse_response(
                {"raw_response": response}
            )
            inputs["raw"] = parsed_response
            return inputs
        
    # ---------------------------------------------
    # STEP 3 → Save to TinyDB
    # ---------------------------------------------
    def save_to_tinydb(self, inputs):
        record = {"product_id": inputs["product_id"], "attributes": inputs["raw"]}
        print("INFO", "Saving to TinyDB...", record)
        self.db.insert(record)
        return inputs

    # ---------------------------------------------
    # STEP 5 → Add summary to Chroma vectorstore
    # ---------------------------------------------
    def add_images_to_vectorstore(self, inputs):
        if "image_path" in inputs or "image_paths" in inputs:
            # Skip vectorstore for UI flow for now
            return
        texts = []
        metadatas = []
        ids = []
        s3_prefix = f"https://rs-apparels.s3.ap-south-1.amazonaws.com/{inputs['product_id']}/cleaned/"

        image_files = self.get_image_files(inputs["input_folder"])

        for i, path in enumerate(image_files):
            view = path.split(".")[0]

            texts.append(f"{view} view of kurti")  # 👈 REQUIRED for RAG
            metadatas.append(
                {
                    "type": path,
                    "view": view,
                    "product_id": inputs["product_id"],
                    "s3_url": s3_prefix + path,
                }
            )
            ids.append(f"img_{i}")
        print("--------------------\nadding images to vector", ids, texts, metadatas)
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    # ---------------------------------------------
    # FULL CHAIN
    # ---------------------------------------------
    def chain(self):
        return (
            RunnableLambda(self.load_image)
            | RunnableLambda(self.extract_attributes)
            | RunnableLambda(self.save_to_tinydb)
        )

    # ---------------------------------------------
    # RUNNER
    # ---------------------------------------------
    def invoke(self, inputs):
        return self.chain().invoke(inputs)
