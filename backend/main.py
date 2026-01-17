from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from PIL import Image
from transformers import AutoModel, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, AutoProcessor
import io
import torch
import torch.nn.functional as F
import os


app = FastAPI()

#Supabase client set up
url = "https://hdlublyjsqrsfzsrcocu.supabase.co"
key = "sb_publishable_OxgcUB5WTh8oMUteqKJIHA_Hqj59R5N"
supabase: Client = create_client(url, key)

#CLIP Setup

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# compute embedding
def compute_image_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.encode_image(inputs["pixel_values"])
    return embedding.squeeze(0).cpu().tolist()  # convert to list for Supabase

