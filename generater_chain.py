import os
import base64
from PIL import Image
from io import BytesIO
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from clip_embeddings import CLIPEmbeddings
from langchain_core.runnables import RunnableLambda
from factory.modal_factory_v2 import ModelFactory
from prompts.get_prompt import get_prompt
from tools.image_util import paths_to_b64urls
from factory.content_bundle import UserContent


class GeneratorChain:
    """
    Agent class for generating kurti images
    """

    def __init__(
        self,
        openai_api_key,
        openrouter_token,
        image_edit_token=None,
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.openrouter_token = openrouter_token
        self.image_edit_token = image_edit_token or os.getenv("SILICONFLOW_API_KEY")
        self.clip_embeddings = CLIPEmbeddings()
        self.initialize_llm()
        self.initialize_prompt()

    def base64_to_image(self, b64_string):
        # decode base64
        image_bytes = base64.b64decode(b64_string)
        # convert to image
        image = Image.open(BytesIO(image_bytes))
        return image

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
            template=get_prompt("generate_prompt_template")
        )

    def get_context(self, inputs):
        context = inputs["design_json"]
        # convert the context into a key value pair string
        context = "\n".join([f"{key}: {value}" for key, value in context.items()])
        print("---------context---------\n", context)
        return {
            "context": context,
            "description": inputs["description"],
            "view": inputs["view"],
            "market_place": inputs["market_place"],
            "feedback": inputs.get("feedback", "")
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

        return final_prompt

    def generate_prompt_chain(self):
        return (
            RunnableLambda(self.get_context)
            | self.prompt
            | self.llm
            | RunnableLambda(self.prompt_parser)
        )

    def generate_prompt(self, inputs):
        result = self.generate_prompt_chain().invoke(inputs)
        return result

    def generate_with_reference(self, prompt, img):
        print("---------generating image with reference---------\n", prompt)
        result = self.client.images.edit(model="gpt-image-1", image=img, prompt=prompt)
        return result.data[0].b64_json

    def generate_with_siliconflow(self, prompt, img_list):
        print("---------generating image with siliconflow---------\n", len(img_list), prompt)
        
        # img_list is expected to be a list of file-like objects from Gradio
        # We need the path or the content. Since Gradio gives file objects, we read them.
        base64_imgs = paths_to_b64urls(img_list)
        return ModelFactory.call_image_edit("image_edit", prompt, base64_imgs)

    def generate_image(self, prompt, reference_images, output_path):
        base64 = self.generate_with_siliconflow(prompt, reference_images)
        print("---------base64 generated successfully---------\n", len(base64))
        image = self.base64_to_image(base64)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        return output_path

    def generate_image_v2(self, prompt, input_paths):
        reference_images = paths_to_b64urls(input_paths)
        bundle = UserContent(system_prompt=prompt, images=reference_images, temperature=0.1)
        base64 = ModelFactory.call_image_edit("image_edit", bundle)
        print("---------base64 generated successfully---------\n", len(base64))
        return base64

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
