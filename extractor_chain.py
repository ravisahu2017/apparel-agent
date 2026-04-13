import json
from tinydb import TinyDB
from prompts.get_prompt import get_prompt
from factory.modal_factory_v2 import ModelFactory
from factory.content_bundle import UserContent
from langchain_core.runnables import RunnableLambda
from langchain_chroma import Chroma
from clip_embeddings import CLIPEmbeddings
from tools.image_util import read_image_files, path_to_base64url

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
        raw_response = inputs.get("extracted_attributes", "")

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
            image_content.append(path_to_base64url(img_path))
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

        prompt = get_prompt("extraction_prompt_v2")
        
        bundle = UserContent(text="Analyze this image and extract the requested attributes.", images=image_content, temperature=0.1)
        inputs["extracted_attributes"] = ModelFactory.call_model("vision", prompt, bundle)
        return inputs

    # ---------------------------------------------
    # FULL CHAIN
    # ---------------------------------------------
    def chain(self):
        return (
            RunnableLambda(self.image_path_handler)
            | RunnableLambda(self.extract_attributes)
            | RunnableLambda(self.parse_response)
        )


    def convert_to_json(self, text):
        prompt = """
        You are a JSON formatter. You have been given a response from a vision model. Convert the response to a valid JSON object.
        """
        schema = {
            "Type of garment": "Kurti", 
            "Silhouette": "", 
            "Patterns": [], 
            "Colors": [], 
            "Sleeves": "", 
            "Top length": "", 
            "Neck design": "", 
            "Border hem details": "", 
            "Notable visual details": "", 
            "Style category": "", 
            "Keywords": []
        }
        prompt = f"You are a fashion data parser. Convert raw text into valid JSON according to this schema: {schema}. Return ONLY the JSON object. No preamble."

        return ModelFactory.call_model("general", prompt, text)


    # ---------------------------------------------
    # RUNNER
    # ---------------------------------------------
    def invoke1(self, inputs):
        for i in [0,1,2]:
            print("INFO", f"Attempt {i+1} starting extraction chain")
            response = self.chain().invoke(inputs)
            if isinstance(response, dict):
                return response
            print("INFO", f"Attempt {i+1} failed, retrying...")

        return None

    def invoke(self, inputs):
        return self.chain().invoke(inputs)
