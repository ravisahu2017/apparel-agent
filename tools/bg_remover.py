import os
import requests
from langchain.tools import tool

REMOVE_BG_BASE_URL = "https://api.remove.bg/v1.0"


@tool(
    "remove_background",
    description="Removes background from an image and saves the result",
    return_direct=False,
)
def remove_background(image_path, output_dir):
    """
    Removes background from the given image and saves result.
    Returns cleaned image path.
    """

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"clean_{filename}")

    API_URL = f"{REMOVE_BG_BASE_URL}/removebg"
    API_KEY = os.getenv("REMOVE_BG_API_KEY")

    with open(image_path, "rb") as image_file:
        response = requests.post(
            API_URL,
            files={"image_file": image_file},
            data={"size": "auto"},
            headers={"X-Api-Key": API_KEY},
        )

    if response.status_code != 200:
        raise Exception(
            f"Background removal failed: {response.status_code} - {response.text}"
        )

    with open(output_path, "wb") as out:
        out.write(response.content)

    return output_path


def account():
    """Check if API key is valid by making a test request"""
    API_URL = f"{REMOVE_BG_BASE_URL}/account"
    API_KEY = os.getenv("REMOVE_BG_API_KEY")
    response = requests.get(API_URL, headers={"X-Api-Key": API_KEY})
    if response.status_code == 200:
        data = response.json()
        return f"API Key is valid. Remaining credits: {data['data']['credits']}"
    else:
        return f"API Key validation failed: {response.status_code} - {response.text}"


if __name__ == "__main__":
    print(account())
