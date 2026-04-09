import os
import requests
from tools.image_util import url_to_base64
from config import MODELS

class ModelFactory:
    @staticmethod
    def call_model(agent_role, prompt, user_content=None):
        config_list = MODELS[agent_role]
        print("INFO", f"Calling model for agent role: {agent_role}")
        for model in config_list:
            try:
                if model["provider"] == "openrouter":
                    return ModelFactory._call_openrouter(model, prompt, user_content)
                elif model["provider"] == "siliconflow":
                    return ModelFactory._call_siliconflow(model, prompt, user_content)
            except Exception as e:
                print("ERROR", f"Failed to use model {model['id']}: {e}")
                continue

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
    def _call_openrouter(model, system_prompt, user_content):
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
    def _call_siliconflow(model, system_prompt, user_content):
        # Your specific SiliconFlow requests logic here
        # Uses os.getenv("SILICONFLOW_API_KEY")
        headers = {
            "Authorization": f"Bearer {os.getenv('SILICONFLOW_API_KEY')}",
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

        print("INFO", f"Calling SiliconFlow with model: {model['id']}")
        response = requests.post(
            "https://api.siliconflow.com/v1/chat/completions",
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
        
    