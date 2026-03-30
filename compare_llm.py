import requests

class CompareByVisionLLM:
    def __init__(self, openrouter_key, openrouter_model="openai/gpt-4o"):
        self.api_key = openrouter_key
        self.openrouter_model = openrouter_model

    def compare(self, raw_image_path, generated_image_path):
        GENERATED_NAME = generated_image_path.split("/")[-1]
        RAW_NAME = raw_image_path.split("/")[-1]
        prompt = f"""
            You are an expert image-comparison AI. You will be provided with a generated image and a raw image.
            You need to compare the kurti worn by the model in the generated image with the raw kurti image and
            highlight the differences in the kurti worn by the model in the generated image with the raw image.

            Provide images are with filenames:
            - Generated image name: {GENERATED_NAME}
            - Raw image name: {RAW_NAME}

            Your job:
            1. Refer to images only by the given filenames.
            2. Provide a detailed visual & semantic comparison.
            3. Highlight differences in:
            - Top length
            - Neck design
            - Sleeve length and design
            - Colors
            - Missing or extra elements
            4. Give a match score between the two.
            5. Provide a detailed explanation of the differences.
            6. Extract the following attributes from {GENERATED_NAME} image only
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
                    {"type": "text", "text": "What are the differences in kurti worn by the model in the generated image compared to the raw image?"},
                    {"type": "image", "image": raw_image_path},
                    {"type": "image", "image": generated_image_path}
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
