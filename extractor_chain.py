import json
from tinydb import TinyDB
from factory import ModelFactory
from langchain_core.runnables import RunnableLambda
from langchain_chroma import Chroma
from clip_embeddings import CLIPEmbeddings
from tools.image_util import read_image_files, map_image_to_openai

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
    def image_path_handler(self, inputs):
        """
        Handles image path input from various sources (Gradio UI, single path, folder)
        Converts images present in paths to base64 format and returns them as a copatible input list for a vision model
        Args:
            inputs: Dictionary containing image paths or folder path
            
        Returns:
            List of image file paths
        """
        # Handle multiple image paths (for Gradio UI), single path, or folder
        if "image_paths" in inputs:
            image_files = inputs["image_paths"]
        elif "image_path" in inputs:
            image_files = [inputs["image_path"]]
        else:
            input_folder = inputs["input_folder"]
            image_files = read_image_files(input_folder)
            image_files = sorted(image_files)

        # Convert images to base64
        image_content = []
        for img_path in image_files:
            image_content.extend(map_image_to_openai(img_path))
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
            image_content: List of image content in base64 format

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
        return inputs["raw"]

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
            RunnableLambda(self.image_path_handler)
            | RunnableLambda(self.extract_attributes)
            | RunnableLambda(self.save_to_tinydb)
        )

    # ---------------------------------------------
    # RUNNER
    # ---------------------------------------------
    def invoke(self, inputs):
        return self.chain().invoke(inputs)
