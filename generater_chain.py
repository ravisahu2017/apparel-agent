import os
import base64
import requests
import uuid
import tempfile
from PIL import Image
from io import BytesIO
from openai import OpenAI
from tinydb import TinyDB, Query
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from clip_embeddings import CLIPEmbeddings
from langchain_core.runnables import RunnableLambda
from factory import ModelFactory
from langchain_chroma import Chroma


class GeneratorChain:
    """
    Agent class for generating kurti images
    """

    def __init__(
        self,
        openai_api_key,
        openrouter_token,
        image_edit_token=None,
        tinydb_path="vision_data.json",
        vectorstore_collection="images",
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.openrouter_token = openrouter_token
        self.image_edit_token = image_edit_token or os.getenv("SILICONFLOW_API_KEY")
        self.clip_embeddings = CLIPEmbeddings()
        self.db = TinyDB(tinydb_path)
        self.vectorstore_collection = vectorstore_collection
        # self.initialize_vectorstore()
        self.initialize_llm()
        self.initialize_prompt()

    def base64_to_image(self, b64_string):
        # decode base64
        image_bytes = base64.b64decode(b64_string)

        # convert to image
        image = Image.open(BytesIO(image_bytes))

        return image

    def initialize_vectorstore(self):
        self.vectorstore = Chroma(
            collection_name=self.vectorstore_collection,
            embedding_function=self.clip_embeddings,
        )
        self.retriever = self.vectorstore.as_retriever()

    def initialize_llm(self):
        self.llm = ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=self.openrouter_token,
            model="nvidia/nemotron-3-super-120b-a12b:free",
            temperature=0.5,
        )

    def initialize_prompt(self):
        self.prompt = PromptTemplate(
            input_variables=["context", "description", "view", "market_place", "feedback"],
            template="""
                You are an expert AI Image Synthesis Engineer for {market_place}.
                Your goal is to create a 'Reference-Guided' prompt. 

                ### STEP 1: RE-ACT ANALYSIS
                - Thought: I must identify the core garment from the context and map it to the user's requested {view}.
                - Action: Create a prompt that anchors the AI to the provided reference image.

                ### FEEDBACK FROM PREVIOUS ATTEMPT:
                {feedback}

                ### STEP 2: GENERATION PROMPT (Return this only)
                [Primary Reference]: Use the attached source image as the structural foundation.
                
                [Actionable Scene]: A professional female fashion model in a {view} pose, high-end {market_place} ecommerce catalog style.
                
                [Garment Integrity Contract]: 
                The model is wearing the EXACT Kurti from the reference image. 
                Specifications to enforce:
                {context}

                Photorealistic, studio lighting, clean white background, 8k resolution. Focus on the high-quality fabric texture of the {view} view. Ensure the embroidery edges and pineapple motifs are sharp and consistent with the reference.

                [Image-to-Image Logic]: 
                Transfer the garment from the reference image onto the model. Maintain the silhouette and fabric drape exactly as shown in the source. 
                {description}.
            """
        )

    def get_context(self, inputs):
        # retrieve attributes from tinydb
        records = self.db.all()
        context = records[0]["attributes"]
        # if single element in context object, then context = context[that single key]
        if len(context) == 1:
            context = context[list(context.keys())[0]]

        # convert the context into a key value pair string
        context = "\n".join([f"{key}: {value}" for key, value in context.items()])

        feedback = inputs.get("feedback", "No previous feedback. This is the first attempt.")

        print("---------context---------\n", context)
        return {
            "context": context,
            "description": inputs["description"],
            "view": inputs["view"],
            "market_place": inputs["market_place"],
            "feedback": feedback,
        }

    def prompt_parser(self, prompt):
        """
        Parse and clean the LLM response to extract only the final prompt

        Args:
            inputs: Dictionary containing the LLM response

        Returns:
            Dictionary with cleaned prompt only
        """
        print("\n---------prompt---------\n", prompt)
        llm_response = prompt

        if not llm_response:
            return {"error": "No LLM response to parse"}

        # Remove common explanatory phrases and prefixes

        response_lines = ""
        try:
            response_lines = llm_response.split("\n")
        except:
            llm_response = llm_response.content
            response_lines = llm_response.split("\n")
        cleaned_lines = []

        # Skip lines that are explanations or headers
        skip_phrases = [
            "Here is the optimized prompt:",
            "Here's the optimized prompt:",
            "Optimized prompt:",
            "Final prompt:",
            "Generated prompt:",
            "Here is your final prompt:",
            "Here's your final prompt:",
            "This prompt is optimized",
            "The optimized prompt is:",
            "Final optimized prompt:",
            "Here is the highly detailed and optimized prompt:",
            "I'll create a highly detailed and optimized prompt",
            "Let me create a highly detailed and optimized prompt",
            "Based on the context",
            "Here's a detailed prompt",
            "Here is a prompt",
            "This prompt",
            "The prompt",
            "Here is your prompt",
            "Here's your prompt",
            "Final prompt for image generation:",
            "Optimized prompt for image generation:",
        ]

        # Process each line
        for line in response_lines:
            line = line.strip()

            # Skip if line contains skip phrases
            if any(phrase.lower() in line.lower() for phrase in skip_phrases):
                continue

            # Skip if line is too short or looks like a header
            if len(line) < 10:
                continue

            # Keep the line if it looks like actual prompt content
            cleaned_lines.append(line)

        # Join the cleaned lines
        final_prompt = "\n".join(cleaned_lines).strip()

        # Additional cleanup - remove any remaining prefixes
        prefixes_to_remove = [
            "Prompt:",
            "Here is the prompt:",
            "Final prompt:",
            "Optimized prompt:",
        ]

        for prefix in prefixes_to_remove:
            if final_prompt.startswith(prefix):
                final_prompt = final_prompt[len(prefix) :].strip()

        return {
            "cleaned_prompt": final_prompt,
            "original_length": len(llm_response),
            "cleaned_length": len(final_prompt),
            "lines_removed": len(response_lines) - len(cleaned_lines),
        }

    def chain(self):
        return (
            RunnableLambda(self.get_context)
            | self.prompt
            | self.llm
            | RunnableLambda(self.prompt_parser)
        )

    def generate_with_reference(self, prompt, img):
        print("---------generating image with reference---------\n", prompt)
        result = self.client.images.edit(model="gpt-image-1", image=img, prompt=prompt)
        return result.data[0].b64_json

    def generate_with_siliconflow(self, prompt, img_list):
        print("---------generating image with siliconflow---------\n", len(img_list), prompt)
        
        # img_list is expected to be a list of file-like objects from Gradio
        # We need the path or the content. Since Gradio gives file objects, we read them.
        # But wait, the user's snippet used open(base_img_path). 
        # In GradioOrchestrator, ref_files is a list of open file handles.
        base64_imgs = []
        for img in img_list:
            try:
                # Get the first image content from the list of file handles
                img.seek(0)
                img_content = img.read()
                base64_img = base64.b64encode(img_content).decode("utf-8")
                base64_imgs.append(base64_img)
            except Exception as e:
                print(f"Error encoding image: {e}")

        img_url = ModelFactory.call_image_edit("image_edit", prompt, base64_imgs)
        
        if not img_url:
            print(f"SiliconFlow response missing image URL: {img_url}")
            return None

        # Download the image to a temp location
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"siliconflow_{uuid.uuid4()}.png")
        
        print(f"Downloading generated image from: {img_url}")
        img_data = requests.get(img_url).content
        
        with open(temp_path, "wb") as f:
            f.write(img_data)
        
        # Convert to base64
        with open(temp_path, "rb") as f:
            final_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Temporary file {temp_path} deleted.")
            
        return final_b64

    def generate_prompt(self, inputs):
        result = self.chain().invoke(inputs)
        return result

    def generate_image(self, prompt, reference_images, output_path):
        base64 = self.generate_with_siliconflow(prompt, reference_images)
        print("---------base64 generated successfully---------\n", base64)
        image = self.base64_to_image(base64)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        return output_path

    def invoke(self, inputs, reference_images, output_path):
        result = self.chain().invoke(inputs)
        # Save result to output_path

        base64 = self.generate_with_reference(
            result.get("cleaned_prompt", "No prompt generated"), reference_images
        )
        print("---------base64 generated successfully---------\n", base64)
        image = self.base64_to_image(base64)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        image.show()
        return output_path
