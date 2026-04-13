import os
import requests
from factory.content_bundle import UserContent
from tools.image_util import url_to_base64
from factory.config import MODELS


class ModelFactory:
    @staticmethod
    def call_model(agent_role, prompt, user_content: UserContent=None):
        config_list = MODELS[agent_role]
        print("INFO", f"Calling model for agent role: {agent_role}")
        response = None
        for model in config_list:
            print(f"INFO ATTEMPT #{config_list.index(model) + 1}", f"Calling {model['provider']} with model: {model['id']}")
            try:
                if model["provider"] == "openrouter":
                    response = ModelFactory._call_openrouter(model, prompt, user_content)
                elif model["provider"] == "siliconflow":
                    response = ModelFactory._call_siliconflow(model, prompt, user_content)
                elif model["provider"] == "google":
                    response = ModelFactory._call_google(model, prompt, user_content)
            except Exception as e:
                print("ERROR", f"Failed to use model {model['id']}: {e}")
                continue
        print("INFO", f"------model response----\n", response)
        return response

    @staticmethod
    def call_image_edit(agent_role, prompt, content_array):
        config_list = MODELS[agent_role]
        print("INFO", f"Calling model for agent role: {agent_role}")
        for model in config_list:
            try:
                if model["provider"] == "openrouter":
                    return ModelFactory._call_image_edit_openrouter(model, prompt, content_array)
                elif model["provider"] == "siliconflow":
                    return ModelFactory._call_image_edit_siliconflow(model, prompt, content_array)
            except Exception as e:
                print("ERROR", f"Failed to use model {model['id']}: {e}")
                continue
    
    @staticmethod
    def _call_openrouter(model, system_prompt, user_content: UserContent):
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
                {"role": "system", "content": system_prompt},
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
    def _call_google(model, system_prompt, user_content: UserContent):
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
            "system_instruction": {"parts": {"text": system_prompt}},
            "contents": [{"role": "user", "parts": user_parts}],
            "generationConfig": {
                "temperature": user_content.temperature,
                "response_mime_type": "application/json" if "json" in system_prompt.lower() else "text/plain"
            }
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        res_json = response.json()

        if "candidates" in res_json:
            return res_json["candidates"][0]["content"]["parts"][0]["text"]
        else:
            error_msg = res_json.get("error", {}).get("message", "Unknown Google API Error")
            print("WARNING", f"Google Model {model['id']} failed: {error_msg}")
            raise Exception(f"Google API Error: {error_msg}")

    @staticmethod
    def _call_siliconflow(model, system_prompt, user_content: UserContent):
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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message_content},
            ],
            "temperature": model.get("temperature", 0.1),
            "response_format": {"type": "json_object"} if "json" in system_prompt.lower() else None
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        res_json = response.json()

        if "choices" in res_json:
            return res_json["choices"][0]["message"]["content"]
        else:
            print("ERROR", f"Model {model['id']} failed: {res_json.get('error')}")
            raise Exception(f"Model {model['id']} failed: {res_json.get('error')}")


    @staticmethod
    def _call_image_edit_openrouter(model, system_prompt, user_content):
        # Your specific OpenRouter requests logic here
        # Uses os.getenv("OPENROUTER_API_KEY")
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        }
        if user_content:
            payload = {
                "model": model["id"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            }
        else:
            payload = {
                "model": model["id"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                ],
            }

        print("INFO", f"Calling OpenRouter with model: {model['id']}")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60, 
        )
        res_json = response.json()
        if "choices" in res_json:
            return res_json["choices"][0]["message"]["content"]
        else:
            print(
                "WARNING", f"Model {model['id']} failed: {res_json.get('error')}"
            )


    @staticmethod
    def _call_image_edit_siliconflow(model, prompt, base64_images):
        url = "https://api.siliconflow.com/v1/images/generations"

        headers = {
            "Authorization": f"Bearer {os.getenv('SILICONFLOW_API_KEY')}",
            "Content-Type": "application/json",
        }

        payload = {
            "prompt": prompt,
            "model": model["id"],
            "image_size": "512x512",
            "images": base64_images,
        }

        print("INFO", f"Calling SiliconFlow with model: {model['id']}")
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            img_url = response.json()["data"][0]["url"]
        else:
            print(f"SiliconFlow Error: {response.text}")
            if os.getenv("ENV") == "dev":
                img_url = "https://rs-apparels.s3.ap-south-1.amazonaws.com/a7a45301-f29b-40b2-abd7-48bd0b1081d9/cleaned/front.png"
            else:
                raise Exception("SiliconFlow response missing image URL")
            

        if not img_url:
            print(f"SiliconFlow response missing image URL: {img_url}")
            if os.getenv("ENV") == "dev":
                img_url = "https://rs-apparels.s3.ap-south-1.amazonaws.com/a7a45301-f29b-40b2-abd7-48bd0b1081d9/cleaned/front.png"
            else:
                raise Exception("SiliconFlow response missing image URL")
        
        return url_to_base64(img_url)
        
    