import os
import json
import requests
import time
from pydantic import BaseModel
from factory.content_bundle import UserContent
from tools.image_util import url_to_base64
from factory.config import MODELS


class ModelFactory:
    @staticmethod
    def call_model(agent_role, user_content: UserContent = None):
        config_list = MODELS[agent_role]
        print("INFO", f"Calling model for agent role: {agent_role}")
        response = None
        for model in config_list:
            print("INFO", f"ATTEMPT #{config_list.index(model) + 1}", f"Calling {model['provider']} with model: {model['id']}")
            try:
                if model["provider"] == "openrouter":
                    response = ModelFactory._call_openrouter(model, user_content)
                elif model["provider"] == "siliconflow":
                    response = ModelFactory._call_siliconflow(model, user_content)
                elif model["provider"] == "google":
                    response = ModelFactory._call_google(model, user_content)
            except Exception as e:
                print("ERROR", f"Failed to use model {model['id']}: {e}")
                continue
        print("INFO", f"------model response----\n", response)
        return response

    @staticmethod
    def call_image_edit(agent_role, user_content: UserContent = None):
        config_list = MODELS[agent_role]
        print("INFO", f"Calling model for agent role: {agent_role}")
        response = None
        for model in config_list:
            print("INFO", f"ATTEMPT #{config_list.index(model) + 1}", f"Calling {model['provider']} with model: {model['id']}")
            try:
                if model["provider"] == "google":
                    response = ModelFactory._call_image_gen_google(model, user_content)
            except Exception as e:
                print("ERROR", f"Failed to use model {model['id']}: {e}")
                continue
        print("INFO", f"------image edit model response----\n", response)
        return response
    
    @staticmethod
    def _call_openrouter(model, user_content: UserContent):
        # Your specific OpenRouter requests logic here
        # Uses os.getenv("OPENROUTER_API_KEY")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        }

        # 1. Build the Multimodal Content Array
        # We start with the user text (instructions for the vision model)
        message_content = [{"type": "text", "text": content.text}]

        # 2. Append the Images in the required b64 format
        if content.has_images():
            for b64 in content.images:
                message_content.append({"type": "image_url", "image_url": {"url": b64}})

        # 3. Construct the Payload
        payload = {
            "model": model["id"],
            "messages": [
                {"role": "system", "content": user_content.system_prompt},
                {"role": "user", "content": message_content}, # This is the array we just built
            ],
            "temperature": user_content.temperature,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        res_json = response.json()

        if "choices" in res_json:
            return res_json["choices"][0]["message"]["content"]
        else:
            print("Error", f"Model {model['id']} failed: {res_json.get('error')}")
            raise Exception(f"Model {model['id']} failed: {res_json.get('error')}")

    @staticmethod
    def _call_google(model, user_content: UserContent):
        """Native call to Google AI Studio (Gemini)"""
        api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")
        # Google uses a different URL structure
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model['id']}:generateContent?key={api_key}"
        
        headers = {"Content-Type": "application/json"}

        # Gemini structure: System instruction is a separate field from contents
        user_parts = []
        
        if user_content.text:
            user_parts.append({"text": user_content.text})
            
        if user_content.has_images():
            for b64 in user_content.images:
                # Extract raw base64 data from data URL format
                if b64.startswith('data:'):
                    # Remove data:image/png;base64, prefix
                    raw_b64 = b64.split(',')[1]
                else:
                    raw_b64 = b64
                    
                user_parts.append({"inline_data": {"mime_type": "image/png", "data": raw_b64}})
        
        payload = {
            "system_instruction": {"parts": {"text": user_content.system_prompt}},
            "contents": [{"role": "user", "parts": user_parts}],
            "generationConfig": {
                "temperature": user_content.temperature,
                "response_mime_type": "application/json" if "json" in user_content.system_prompt.lower() else "text/plain"
            }
        }

        response = requests.post(
            url=url, 
            headers=headers, 
            json=payload, 
            timeout=60*5
        )
        res_json = response.json()
        print("INFO _call_google api response:\n", res_json)
        if "candidates" in res_json:
            return res_json["candidates"][0]["content"]["parts"][0]["text"]
        else:
            error_msg = res_json.get("error", {}).get("message", "Unknown Google API Error")
            print("WARNING", f"Google Model {model['id']} failed: {error_msg}")
            raise Exception(f"Google API Error: {error_msg}")

    @staticmethod
    def _call_siliconflow(model, user_content: UserContent):
        # Your specific SiliconFlow requests logic here
        # Uses os.getenv("SILICONFLOW_API_KEY")
        headers = {
            "Authorization": f"Bearer {os.getenv('SILICONFLOW_API_KEY')}",
            "Content-Type": "application/json",
        }
        url = "https://api.siliconflow.com/v1/chat/completions"

        # Format 1: Chat/Vision (DeepSeek-VL, Qwen-VL)
        # SiliconFlow follows the OpenAI multimodal standard
        message_content = [{"type": "text", "text": user_content.text}]
        
        if user_content.has_images():
            for b64 in user_content.images:
                message_content.append({"type": "image_url", "image_url": {"url": b64}})


        payload = {
            "model": model["id"],
            "messages": [
                {"role": "system", "content": user_content.system_prompt},
                {"role": "user", "content": message_content},
            ],
            "temperature": model.get("temperature", 0.1),
            "response_format": {"type": "json_object"} if "json" in user_content.system_prompt.lower() else None
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        res_json = response.json()

        if "choices" in res_json:
            return res_json["choices"][0]["message"]["content"]
        else:
            print("ERROR", f"Model {model['id']} failed: {res_json.get('error')}")
            raise Exception(f"Model {model['id']} failed: {res_json.get('error')}")


    @staticmethod
    def _call_image_gen_google(model, content: UserContent):
        """
        Native Google AI Studio call for Multimodal Image Generation.
        Uses reference images from UserContent to guide the output.
        """
        api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")
        # Using v1beta for access to multimodal 'predict' features
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model['id']}:generateContent?key={api_key}"
        
        headers = {"Content-Type": "application/json"}

        # 1. Build the Multimodal contents array
        # Gemini 3 treats image generation as a text + image input/output task

        user_parts = []
        if content.text:
            user_parts.append({"text": content.text})
        
        if content.has_images():
            for i, b64 in enumerate(content.images):
                # Extract raw base64 data from data URL format
                if b64.startswith('data:'):
                    # Remove data:image/png;base64, prefix
                    raw_b64 = b64.split(',')[1]
                else:
                    raw_b64 = b64

                user_parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": raw_b64
                    }
                })

        # 2. Build the payload
        # The prompt should ideally refer to the images as [1], [2], etc.
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": user_parts
                }
            ],
            "generationConfig": {
                # This tells Gemini to output both TEXT and IMAGE
                "responseModalities": ["TEXT", "IMAGE"],
                "candidateCount": 1,
                "imageConfig": {
                    "aspectRatio": "1:1",
                    "imageSize": "1K"
                }
            }
        }

        # Add system instruction if provided
        if content.system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": content.system_prompt}]}

        print("INFO", f"Calling Google Multimodal Gen: {model['id']} with {len(content.images)} refs")
        
        # Retry logic with exponential backoff for rate limiting
        max_retries = 3
        base_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                
                # Handle rate limiting specifically
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"INFO", f"Rate limited. Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        raise Exception(f"Rate limit exceeded after {max_retries} attempts")
                
                response.raise_for_status()
                res_json = response.json()
                
                # Parse the response according to the new format
                if "candidates" in res_json and len(res_json["candidates"]) > 0:
                    candidate = res_json["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        for part in candidate["content"]["parts"]:
                            if "inlineData" in part and "data" in part["inlineData"]:
                                # Return the base64 string of the generated image
                                return part["inlineData"]["data"]
                
                # If we get here, no image was found in the response
                print("ERROR", f"No image in response: {res_json}")
                raise Exception(f"Google Imagen API Error: No image generated - {res_json}")
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"INFO", f"Request failed. Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(f"Image Generation failed after {max_retries} attempts: {e}")
            except Exception as e:
                print("ERROR", f"Image Generation failed: {e}")
                if os.getenv("ENV") == "dev":
                    # Return a fallback from your S3 for dev testing
                    return "DEVELOPMENT_MODE_MOCK_IMAGE_BASE64"
                raise e