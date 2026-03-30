from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
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

        # Ensure we get the tensor and convert to numpy
        if hasattr(emb, 'pooler_output'):
            # If it's a model output object, get the pooler_output
            tensor = emb.pooler_output
        else:
            # If it's already a tensor
            tensor = emb
            
        return tensor.detach().cpu().numpy().flatten()

    def embed_documents(self, docs):
        return [self.embed_query(doc) for doc in docs]

    def embed_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)

        return emb[0].numpy()

    def similarity(self, img1, img2):
        v1 = self.embed_image(img1)
        v2 = self.embed_image(img2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))