import os
import requests
from factory.content_bundle import UserContent
from tools.image_util import url_to_base64
from factory.config import MODELS

class ImageEdit:

    @staticmethod
    def _call_image_edit_google(model, system_prompt, content: UserContent):
        """
        Native Google AI Studio call using Gemini 2.5 Flash.
        Handles multi-image input (up to 4 images) to synthesize a single output instruction or DNA.
        """
        api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")
        # Ensure we are using the v1beta endpoint for 2.5 Flash features
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model['id']}:generateContent?key={api_key}"
        
        headers = {"Content-Type": "application/json"}

        # Build parts array starting with the text instructions
        user_parts = [{"text": content.text if content.text else "Analyze these apparel images."}]

        # Add all 4 images from the bundle
        if content.has_images():
            for b64_data in content.images:
                user_parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": b64_data
                    }
                })

        payload = {
            "system_instruction": {
                "parts": {"text": system_prompt}
            },
            "contents": [
                {
                    "role": "user",
                    "parts": user_parts
                }
            ],
            "generationConfig": {
                "temperature": model.get("temperature", 0.1),
                "response_mime_type": "application/json" if "json" in system_prompt.lower() else "text/plain"
            }
        }

        print("INFO", f"Calling Google Gemini 2.5 Flash with {len(content.images)} images")
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            res_json = response.json()
            
            if "candidates" in res_json:
                return res_json["candidates"][0]["content"]["parts"][0]["text"]
            else:
                raise Exception(f"Google API Error: {res_json}")
                
        except Exception as e:
            print("ERROR", f"Gemini 2.5 Flash failed: {e}")
            # Fallback to dev image if in dev environment
            if os.getenv("ENV") == "dev":
                return "DEVELOPMENT_MODE_MOCK_RESPONSE"
            raise e


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
        
