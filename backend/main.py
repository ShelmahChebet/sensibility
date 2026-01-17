from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPModel
import io
import torch
import torch.nn.functional as F
import os


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#Supabase client set up
url = "https://hdlublyjsqrsfzsrcocu.supabase.co"
key = "sb_publishable_OxgcUB5WTh8oMUteqKJIHA_Hqj59R5N"
supabase: Client = create_client(url, key)

#CLIP Setup

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# compute embedding
def compute_image_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_embedding = image_features.squeeze(0)
        image_embedding = image_embedding / image_embedding.norm(p=2)  # normalize

    return image_embedding.cpu().detach().numpy().astype(float).tolist()

#Track event data
def track_event(user_id: str, event_name: str, properties: dict = {}):
    supabase.table("events").insert({
        "user_id": user_id,
        "event_name": event_name,
        "properties": properties
    }).execute()

def get_or_create_user_profile(user_id: str):
    resp = supabase.table("user_preferences").select("*").eq("user_id", user_id).execute()

    if len(resp.data) == 0:
        profile = {
            "user_id": user_id,
            "preferred_styles": [],
            "disliked_categories": [],
            "formality_score": 0.5
        }
        supabase.table("user_preferences").insert(profile).execute()
        return profile

    return resp.data[0]

def update_preferences(user_id, event_name, properties):
    prefs = supabase.table("user_preferences").select("*").eq("user_id", user_id).single().execute().data

    if event_name == "outfit_skipped":
        category = properties.get("category")
        prefs["disliked_categories"].append(category)

    if event_name == "outfit_clicked":
        prefs["formality_score"] = min(1.0, prefs["formality_score"] + 0.05)

    supabase.table("user_preferences").update(prefs).eq("user_id", user_id).execute()

# Upload into wardrobe endpoint
@app.post("/upload-to-wardrobe/")
async def upload_wardrobe(file: UploadFile = File(...), user_id: str = "test_user", category: str = "top"):
    
    profile = get_or_create_user_profile(user_id)

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
    
    # Track event
    track_event(
        user_id,
        "upload_item",
        {
            "category": category,
            "filename": file.filename
        }
    )
    
    return {"status": "success", "filename": file.filename}

@app.post("/suggest_outfits")
async def suggest_outfits(prompt: str, user_id: str = "test_user"):
    
    profile = get_or_create_user_profile(user_id)

    
    #Track event
    track_event(
        user_id,
        "request_recommendation",
        {"prompt": prompt}
    )

    text_input = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_input)
        text_embedding = text_features.squeeze(0)
        text_embedding = text_embedding / text_embedding.norm(p=2)  # normalize
        text_embedding = text_embedding.cpu().detach()
    
    # Retrieve embeddings from Supabase
    resp = supabase.table("wardrobe_items").select("*").eq("user_id", user_id).execute()
    items = resp.data
    
    # Compute similarities
    results = []
    for item in items:
        item_embedding = torch.tensor(item["embedding"])
        item_embedding = item_embedding / item_embedding.norm(p=2)  # normalize
        similarity = F.cosine_similarity(text_embedding, item_embedding, dim=0)

        # convert cosine similarity (-1 to 1) → percentage (0 to 100)
        percentage = float((similarity + 1) / 2 * 100)

        results.append({
            "filename": item["filename"],
            "score": round(percentage, 2)
        })
    
    # Sort top 5
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"prompt": prompt, "matches": results[:5]}

@app.post("/feedback")
async def outfit_feedback(
    user_id: str,
    filename: str,
    action: str,  # "clicked" or "skipped"
    category: str
):
    track_event(
        user_id,
        f"outfit_{action}",
        {"filename": filename, "category": category}
    )

    update_preferences(user_id, f"outfit_{action}", {"category": category})

    return {"status": "recorded"}