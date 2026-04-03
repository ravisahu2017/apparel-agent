import requests
import base64


class CompareByVisionLLM:
    model_priority = ["nvidia/nemotron-nano-12b-v2-vl:free", "anthropic/claude-3-haiku"]

    def __init__(self, openrouter_key, openrouter_model="openai/gpt-4o"):
        self.api_key = openrouter_key
        self.openrouter_model = openrouter_model

    def encode(self, path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    def compare(self, raw_image_path, generated_image_path):
        prompt = """
            You are an Apparel Quality Control Expert. Compare the 'Original Product' with the 'Generated Fashion Model'.
            
            Verification Criteria:
            1. Does the generated image contain the whole body?
            2. Is the sleeve length (Short/3-4th/Full) identical to the original?
            3. Is the neck style (V-neck, Mandarin, Round) identical?
            4. Is the repeating pattern/print exactly the same as the original?
            
            Return a JSON response only:
            {
            "is_passed": boolean,
            "score": 1-10,
            "issues": ["list of discrepancies"],
            "feedback_for_regeneration": "Specific instructions to fix the image"
            }
            """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.openrouter_model,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What are the differences in kurti worn by the model in the generated image compared to the raw image?",
                        },
                        {
                            "type": "text",
                            "text": "The FIRST image is the 'Original Product'. The SECOND image is the 'Generated Result'.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self.encode(raw_image_path)}"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self.encode(generated_image_path)}"
                            },
                        },
                    ],
                },
            ],
        }

        for model_id in self.model_priority:
            payload["model"] = model_id
            try:
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
                        "WARNING", f"Model {model_id} failed: {res_json.get('error')}"
                    )
            except Exception as e:
                print("ERROR", f"Failed to use model {model_id}: {e}")
                continue
