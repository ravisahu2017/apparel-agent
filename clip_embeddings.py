from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

class CLIPEmbeddings:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def embed_query(self, query):
        # Handle image query
        if query.endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(query).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")

            with torch.no_grad():
                emb = self.model.get_image_features(**inputs)
        else:
            # Handle text query
            inputs = self.processor(text=[query], return_tensors="pt", padding=True)

            with torch.no_grad():
                emb = self.model.get_text_features(**inputs)

        return emb[0].numpy()

    def embed_documents(self, docs):
        return [self.embed_query(doc) for doc in docs]

