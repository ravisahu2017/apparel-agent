import requests
import base64
import json
import io
from openai import OpenAI
from tinydb import TinyDB, Query
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda


class GeneratorChainV2:
    model_priority = [
        "black-forest-labs/FLUX-1.1-pro-Ultra",
    ]

    def __init__(
        self, openrouter_token, image_edit_token, tinydb_path="vision_data.json"
    ):
        # We use OpenRouter for the LLM (Free Models)
        self.openrouter_token = openrouter_token
        self.image_edit_token = image_edit_token
        self.db = TinyDB(tinydb_path)

        self.initialize_llm()
        self.initialize_prompt()

    def initialize_llm(self):
        # Using a reliable free model on OpenRouter
        self.llm = ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=self.openrouter_token,
            model="anthropic/claude-3-haiku",  # Upgraded free model for 2026
            temperature=0.3,
        )

    def initialize_prompt(self):
        self.prompt = PromptTemplate(
            input_variables=["context", "description", "view", "market_place"],
            template="""
                You are an expert fashion stylist for {market_place}.
                Create a professional image generation prompt for a female model wearing this kurti.
                
                Product Details: {context}
                User Request: {description}
                View: {view}

                The prompt must be descriptive (fabric texture, lighting, background).
                Return ONLY the prompt text.
            """,
        )

    def get_context(self, inputs):
        # retrieve attributes from tinydb
        records = self.db.all()
        context = records[0]["attributes"]
        if "raw" in context:
            context = context["raw"]

        print("---------context---------\n", context)
        return {
            "context": context,
            "description": inputs["description"],
            "view": inputs["view"],
            "market_place": inputs["market_place"],
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

        print("\n---------cleaned prompt---------\n", final_prompt)

        return {
            "cleaned_prompt": final_prompt,
            "original_length": len(llm_response),
            "cleaned_length": len(final_prompt),
            "lines_removed": len(response_lines) - len(cleaned_lines),
        }

    def generate_with_reference1(self, prompt, base_img_path, mask_img_path=None):
        """
        Updated Free Generation Logic for 2026.
        Uses Pixazo (Unified Free API) as the primary and SiliconFlow as backup.
        """
        print(f"--- Starting Image Edit ---")

        if self.image_edit_token:
            try:
                print("Attempting SiliconFlow (Direct SDXL)...")
                # Corrected 2026 endpoint for general generation/editing
                url = "https://api.siliconflow.com/v1/images/generations"

                headers = {
                    "Authorization": f"Bearer {self.image_edit_token}",
                    "Content-Type": "application/json",
                }

                payload = {
                    "model": "Qwen/Qwen-Image-Edit",
                    "prompt": prompt,
                    "image_size": "512x512",
                    "image": base64.b64encode(open(base_img_path, "rb").read()).decode(
                        "utf-8"
                    ),
                }

                response = requests.post(url, json=payload, headers=headers)

                # SAFE PARSING to avoid 'Extra Data' error
                if response.status_code == 200:
                    return response.json()["data"][0]["url"]
                else:
                    print(f"SiliconFlow Error: {response.text}")
            except Exception as e:
                print(f"SiliconFlow failed: {e}")

        return "Error: All free generation endpoints failed."

    def generate_with_reference(self, prompt, base_img_path):
        url = "https://api.segmind.com/v1/qwen-image-edit"
        payload = {
            "image": base64.b64encode(open(base_img_path, "rb").read()).decode("utf-8"),
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, distorted",
            "samples": 1,
            "steps": 25,
        }
        headers = {"x-api-key": self.image_edit_token}
        response = requests.post(url, json=payload, headers=headers)
        # Returns base64 or URL depending on settings
        return response.json()

    def chain(self):
        return (
            RunnableLambda(self.get_context)
            | self.prompt
            | self.llm
            | RunnableLambda(self.prompt_parser)
        )

    def invoke(self, inputs, base_img_path):
        # 1. Run LLM to get the optimized prompt
        chain_output = self.chain().invoke(inputs)
        final_prompt = chain_output["cleaned_prompt"]

        # 2. Run Image Generation/Edit
        return self.generate_with_reference(
            prompt=final_prompt, base_img_path=base_img_path
        )
