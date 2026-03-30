import base64
import requests

class VisionLLMExtractor:
    def __init__(self, openrouter_key, openrouter_model="openai/gpt-4o"):
        self.api_key = openrouter_key
        self.openrouter_model = openrouter_model

    def extract(self, image_path):
        img_b64 = base64.b64encode(open(image_path, "rb").read()).decode()

        prompt = """
            You are a fashion vision model. You have been given a kurti in the image. Analyze the kurti item in the image
            and extract the following attributes:
                - Is full body of the model visible?
                - Is full kurti visible?
                - Sleeve length
                - Kurti length (hips/knees/ankle/thighs)
                - Neck style
                Return JSON only.
            """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.openrouter_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this image and extract the requested attributes."},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"}
                ]}
            ]
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )

        result = response.json()
        print("INFO", "OpenRouter API response:", result)
        
        if "error" in result:
            print("ERROR", f"OpenRouter API error: {result['error']}")
            return {"error": result['error']}
        
        if "choices" not in result:
            print("ERROR", f"Unexpected response format: {result}")
            return {"error": "Unexpected response format"}
        
        raw_response = result["choices"][0]["message"]["content"]
        return raw_response


    def compare_images(raw, generated, question):
        raw_b64 = encode(raw)
        generated_b64 = encode(generated)

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": 
                        f"Compare these two images and answer the question:\n{question}\n"
                        "Respond in JSON with fields: answer, reasoning."},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{raw_b64}"},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{generated_b64}"}
                    ]
                }
            ]
        }

        return response.choices[0].message["content"]
        