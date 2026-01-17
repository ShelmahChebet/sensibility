"""
Wardrobe AI Recommendation System
A FastAPI application that uses CLIP (Vision-Language Model) to suggest outfit combinations
based on user text descriptions and their personal wardrobe items.
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import io
import torch
import torch.nn.functional as F
import os
from dotenv import load_dotenv
from llm import analyze_user_style, explain_recommendations, infer_item_style


# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# CORS middleware: Allow requests from any origin (tighten for production)
# This is necessary for frontend-to-backend communication across different domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client with credentials from environment variables
# SUPABASE_URL and SUPABASE_KEY are set in .env file
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# ============================================================================
# CLIP Model Setup - Vision-Language Model for Image/Text Embeddings
# ============================================================================
# CLIP is trained to understand both images and text in the same embedding space.
# This allows us to compare an image with a text description to find matches.

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def compute_image_embedding(image: Image.Image):
    """
    Generate a CLIP embedding for an image.
    
    Args:
        image: PIL Image object (RGB format)
    
    Returns:
        List of floats representing the image's embedding vector
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_embedding = image_features.squeeze(0)
        image_embedding = image_embedding / image_embedding.norm(p=2)  # normalize to unit vector

    return image_embedding.cpu().detach().numpy().astype(float).tolist()

# ============================================================================
# Event Tracking and User Preference Management
# ============================================================================

def track_event(user_id: str, event_name: str, properties: dict = {}):
    """
    Log user interactions for analytics and preference learning.
    
    Args:
        user_id: Unique identifier for the user
        event_name: Type of event (e.g., 'upload_item', 'outfit_clicked')
        properties: Additional metadata about the event
    """
    supabase.table("events").insert({
        "user_id": user_id,
        "event_name": event_name,
        "properties": properties
    }).execute()

def get_or_create_user_profile(user_id: str, preferred_styles: list = [], category_formality: dict = {}):
    """
    Retrieve a user's preference profile, or create a new one if it doesn't exist.
    
    Args:
        user_id: Unique identifier for the user
    
    Returns:
        Dictionary containing user preferences
    """
    resp = supabase.table("user_preferences").select("*").eq("user_id", user_id).execute()

    if len(resp.data) == 0:
        profile = {
        "user_id": user_id,
        "preferred_styles": preferred_styles,
        "disliked_categories": [],
        "category_formality": category_formality,
        "formality_score": 0.5
        }
        supabase.table("user_preferences").insert(profile).execute()
        return {"status": "profile_created", "profile": profile}

    return resp.data[0]

def update_preferences(user_id, event_name, properties):
    """
    Update user preferences based on their interactions.
    
    - 'outfit_skipped': Adds category to disliked list
    - 'outfit_clicked': Increases formality score (user likes formal items)
    
    Args:
        user_id: User identifier
        event_name: Type of interaction
        properties: Event details
    """
    prefs = supabase.table("user_preferences").select("*").eq("user_id", user_id).single().execute().data

    if event_name == "outfit_skipped":
        category = properties.get("category")
        prefs["disliked_categories"].append(category)

    if event_name == "outfit_clicked":
        prefs["formality_score"] = min(1.0, prefs["formality_score"] + 0.05)

    supabase.table("user_preferences").update(prefs).eq("user_id", user_id).execute()
    
def apply_profile_weights(matches, user_preferences):
    weighted_results = []
    for match in matches:
        score = match["score"]

        # Penalize disliked categories
        if match.get("category") in user_preferences["disliked_categories"]:
            score *= 0.7  # 30% penalty

        # Boost preferred styles (if you have style info for items)
        if match.get("style") in user_preferences["preferred_styles"]:
            score *= 1.1  # 10% boost

        # Adjust based on formality (if items have a formality score)
        if "formality" in match:
            score *= 0.5 + 0.5 * user_preferences["formality_score"]  # scale 0.5-1

        weighted_results.append({**match, "weighted_score": round(score, 2)})

    # Sort by weighted score
    weighted_results.sort(key=lambda x: x["weighted_score"], reverse=True)
    return weighted_results


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/upload-to-wardrobe/")
async def upload_wardrobe(file: UploadFile = File(...), user_id: str = "test_user", category: str = "top"):
    """
    Upload a clothing item image to a user's wardrobe.
    
    The image is processed to create a CLIP embedding, which is stored alongside
    the image metadata for later similarity matching.
    
    Args:
        file: Image file to upload (JPEG, PNG, etc.)
        user_id: User identifier (default: "test_user")
        category: Clothing category (e.g., 'top', 'bottom', 'dress')
    
    Returns:
        Status message with filename
    """
    
    profile = get_or_create_user_profile(user_id)

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    style = infer_item_style(image)
    embedding = compute_image_embedding(image)

    
    # Store the wardrobe item with its embedding in Supabase
    supabase.table("wardrobe_items").insert({
        "filename": file.filename,
        "embedding": embedding,
        "user_id": user_id,
        "category": category,
        "style": style
    }).execute()
    
    # Record this upload event for analytics
    track_event(
        user_id,
        "upload_item",
        {
            "category": category,
            "filename": file.filename,
            "style": style
        }
    )
    
    return {"status": "success", "filename": file.filename}

@app.post("/suggest_outfits")
async def suggest_outfits(prompt: str, user_id: str = "test_user"):
    """
    Suggest outfit items from a user's wardrobe based on a text description.
    
    Process:
    1. Convert text prompt to CLIP embedding
    2. Compare against all user's wardrobe item embeddings
    3. Rank items by cosine similarity score
    4. Use Claude to analyze user style from past events
    5. Use GPT-4o to generate personalized explanation
    
    Args:
        prompt: Text description (e.g., "business casual outfit for work")
        user_id: User identifier (default: "test_user")
    
    Returns:
        Top 5 matching items with scores and AI-generated explanation
    """

    # Log recommendation request
    track_event(
        user_id,
        "request_recommendation",
        {"prompt": prompt}
    )

    # Step 1: Generate text embedding from the user's prompt
    text_input = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_input)
        text_embedding = text_features.squeeze(0)
        text_embedding = text_embedding / text_embedding.norm(p=2)  # normalize
        text_embedding = text_embedding.cpu().detach()
        
    # Step 1b: Fetch all wardrobe items for the user
    resp = supabase.table("wardrobe_items").select("*").eq("user_id", user_id).execute()
    items = resp.data
    
    # Step 1: Fetch recent events
    events = (
        supabase.table("events")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(20)
        .execute()
        .data
    )

    # Step 2: Analyze user style with Claude
    style_insights = analyze_user_style(
        events=events,
        wardrobe_summary=[item["category"] for item in items]
    )

    # Step 3: Compute similarity scores between text prompt and wardrobe items
    results = []
    profile = get_or_create_user_profile(user_id)
    formality_map = profile.get("category_formality", {})

    for item in items:
        item_embedding = torch.tensor(item["embedding"])
        item_embedding = item_embedding / item_embedding.norm(p=2)
        similarity = F.cosine_similarity(text_embedding, item_embedding, dim=0)
        results.append({
            "filename": item["filename"],
            "category": item["category"],
            "formality": formality_map.get(item["category"], 0.5),
            "score": float((similarity + 1) / 2 * 100)
        })

    # Step 4: Apply profile weights (penalize disliked, boost preferred, adjust formality)
    weighted_results = apply_profile_weights(results, style_insights)

    # Step 5: Take top 5 weighted results
    top_matches = weighted_results[:5]

    # Step 6: Generate explanations using GPT-4o
    explanation = explain_recommendations(
        prompt=prompt,
        matches=top_matches,
        user_profile=style_insights
    )

    return {
        "prompt": prompt,
        "matches": top_matches,
        "explanation": explanation,
        "models_used": {
            "behavior_analysis": "Anthropic Claude",
            "explanation": "OpenAI GPT-4o"
        }
    }



@app.post("/feedback")
async def outfit_feedback(
    user_id: str,
    filename: str,
    action: str,  # "clicked" or "skipped"
    category: str
):
    """
    Record user feedback on outfit suggestions.
    
    This feedback helps personalize future recommendations:
    - 'clicked': User liked the suggestion
    - 'skipped': User disliked the suggestion
    
    Args:
        user_id: User identifier
        filename: The clothing item's filename
        action: User's action ('clicked' or 'skipped')
        category: Clothing category
    
    Returns:
        Status confirmation
    """
    # Record the feedback event
    track_event(
        user_id,
        f"outfit_{action}",
        {"filename": filename, "category": category}
    )

    # Update user preferences based on their feedback
    update_preferences(user_id, f"outfit_{action}", {"category": category})

    return {"status": "recorded"}