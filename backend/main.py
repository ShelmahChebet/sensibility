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
from outfit_curator import generate_outfits, format_outfit_response


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
async def suggest_outfits(prompt: str, user_id: str = "test_user", num_outfits: int = 3):
    """
    Suggest complete outfit combinations from user's wardrobe based on text description.
    
    Process:
    1. Convert text prompt to CLIP embedding
    2. Fetch all user's wardrobe items
    3. Analyze user style preferences with Claude
    4. Generate complete outfit combinations (top + bottom + accessories)
    5. Score outfits based on compatibility and prompt match
    6. Use GPT-4o to explain outfit choices
    
    Args:
        prompt: Text description (e.g., "business casual outfit for work")
        user_id: User identifier (default: "test_user")
        num_outfits: Number of outfit suggestions (default: 3)
    
    Returns:
        Complete outfit combinations with AI-generated explanations
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
        
    # Step 2: Fetch all wardrobe items for the user
    resp = supabase.table("wardrobe_items").select("*").eq("user_id", user_id).execute()
    items = resp.data
    
    if not items:
        return {
            "error": "No wardrobe items found",
            "message": "Please upload some clothing items first!"
        }
    
    # Step 3: Fetch recent events and analyze user style
    events = (
        supabase.table("events")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(20)
        .execute()
        .data
    )

    style_insights = analyze_user_style(
        events=events,
        wardrobe_summary=[item["category"] for item in items]
    )

    # Step 4: Get user profile for formality preferences
    profile = get_or_create_user_profile(user_id)
    formality_map = profile.get("category_formality", {})

    # Add formality scores to items
    for item in items:
        item["formality"] = formality_map.get(item["category"], 0.5)

    # Step 5: Generate complete outfit combinations
    outfits = generate_outfits(
        items=items,
        text_embedding=text_embedding,
        user_preferences=style_insights,
        clip_model=model,
        clip_processor=processor,
        device=device,
        num_outfits=num_outfits
    )

    if not outfits:
        return {
            "error": "Could not generate outfits",
            "message": "Try uploading more diverse clothing items (tops, bottoms, dresses)"
        }

    # Step 6: Format outfits for response
    formatted_outfits = format_outfit_response(outfits)

    # Step 7: Generate AI explanation for the outfit suggestions
    explanation = explain_recommendations(
        prompt=prompt,
        matches=formatted_outfits,
        user_profile=style_insights
    )

    return {
        "prompt": prompt,
        "num_outfits_generated": len(formatted_outfits),
        "outfits": formatted_outfits,
        "user_style_profile": style_insights,
        "explanation": explanation,
        "models_used": {
            "outfit_generation": "CLIP Vision-Language Model",
            "behavior_analysis": "Anthropic Claude",
            "explanation": "OpenAI GPT-4o"
        }
    }


@app.post("/rate_outfit")
async def rate_outfit(
    user_id: str,
    outfit_number: int,
    rating: int,  # 1-5 stars
    items: list  # List of item filenames in the outfit
):
    """
    Let users rate outfit suggestions to improve future recommendations.
    
    Args:
        user_id: User identifier
        outfit_number: Which outfit suggestion (1, 2, or 3)
        rating: User's rating (1-5 stars)
        items: List of item filenames in the outfit
    
    Returns:
        Confirmation message
    """
    track_event(
        user_id,
        "rate_outfit",
        {
            "outfit_number": outfit_number,
            "rating": rating,
            "items": items
        }
    )
    
    # Update preferences based on rating
    if rating >= 4:  # Positive feedback
        profile = supabase.table("user_preferences").select("*").eq("user_id", user_id).single().execute().data
        
        # Slightly increase formality preference if high-rated outfit was formal
        # This is a simple heuristic - could be more sophisticated
        profile["formality_score"] = min(1.0, profile["formality_score"] + 0.02)
        
        supabase.table("user_preferences").update(profile).eq("user_id", user_id).execute()
    
    return {
        "status": "rating recorded",
        "message": f"Thanks for rating outfit #{outfit_number}!"
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