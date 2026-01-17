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

# Upload into wardrobe endpoint
@app.post("/upload-to-wardrobe/")
async def upload_wardrobe(file: UploadFile = File(...), user_id: str = "test_user", category: str = "top"):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    embedding = compute_image_embedding(image)
    
    # Insert into Supabase
    supabase.table("wardrobe_items").insert({
        "filename": file.filename,
        "embedding": embedding,
        "user_id": user_id,
        "category": category
    }).execute()
    
    return {"status": "success", "filename": file.filename}

@app.post("/suggest_outfits")
async def suggest_outfits(prompt: str, user_id: str = "test_user"):

    text_input = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_input["input_ids"]).squeeze(0)
    
    # Retrieve embeddings from Supabase
    resp = supabase.table("wardrobe_items").select("*").eq("user_id", user_id).execute()
    items = resp.data
    
    # Compute similarities
    results = []
    for item in items:
        item_embedding = torch.tensor(item["embedding"])
        similarity = F.cosine_similarity(text_embedding, item_embedding, dim=0)
        results.append({"filename": item["filename"], "score": float(similarity)})
    
    # Sort top 5
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"prompt": prompt, "matches": results[:5]}
