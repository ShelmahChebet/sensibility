"""
Wardrobe AI Recommendation System
A FastAPI application that uses CLIP (Vision-Language Model) to suggest outfit combinations
based on user text descriptions and their personal wardrobe items.
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from PIL import Image
from supabase_auth import BaseModel
from transformers import CLIPProcessor, CLIPModel
import io
import torch
import torch.nn.functional as F
import os
from dotenv import load_dotenv
from llm import analyze_user_style, explain_recommendations, infer_item_style
from outfit_curator import generate_outfits, format_outfit_response
from pydantic import BaseModel
from typing import List

class RateOutfitRequest(BaseModel):
    user_id: str
    outfit_number: int
    rating: int  # 1-5 stars
    items: List[str]  # List of item filenames
    outfit_type: str = "separates"  # "dress" or "separates"

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

def update_preferences_from_rating(user_id: str, rating: int, outfit_items: list):
    """
    Update user preferences based on outfit ratings.
    Learns what styles and combinations the user likes.
    
    Args:
        user_id: User identifier
        rating: Rating from 1-5
        outfit_items: List of items in the rated outfit
    """
    # Get current preferences
    prefs = supabase.table("user_preferences").select("*").eq("user_id", user_id).single().execute().data
    
    # Extract styles from the outfit items
    outfit_styles = []
    outfit_categories = []
    total_formality = 0
    
    for item in outfit_items:
        # Fetch full item details
        item_data = supabase.table("wardrobe_items").select("*").eq("filename", item).eq("user_id", user_id).single().execute().data
        
        if item_data:
            style = item_data.get("style", "")
            category = item_data.get("category", "")
            
            if style and style != "unknown":
                outfit_styles.append(style)
            if category:
                outfit_categories.append(category)
            
            # Get formality for this category
            formality = prefs.get("category_formality", {}).get(category, 0.5)
            total_formality += formality
    
    avg_formality = total_formality / len(outfit_items) if outfit_items else 0.5
    
    # Update based on rating
    if rating >= 4:  # Positive rating (4-5 stars)
        # Add liked styles to preferred_styles
        for style in outfit_styles:
            if style not in prefs["preferred_styles"]:
                prefs["preferred_styles"].append(style)
        
        # Adjust formality score towards this outfit's formality
        current_formality = prefs.get("formality_score", 0.5)
        # Move 10% towards the rated outfit's formality
        new_formality = current_formality * 0.9 + avg_formality * 0.1
        prefs["formality_score"] = min(1.0, max(0.0, new_formality))
        
    elif rating <= 2:  # Negative rating (1-2 stars)
        # Add categories to disliked if consistently rated low
        for category in outfit_categories:
            # Check if this category has been rated low multiple times
            low_ratings = supabase.table("events").select("*").eq("user_id", user_id).eq("event_name", "rate_outfit").execute().data
            
            category_low_count = 0
            for event in low_ratings:
                if event.get("properties", {}).get("rating", 5) <= 2:
                    event_items = event.get("properties", {}).get("items", [])
                    event_categories = []
                    for item_filename in event_items:
                        item_info = supabase.table("wardrobe_items").select("category").eq("filename", item_filename).eq("user_id", user_id).execute().data
                        if item_info:
                            event_categories.extend([i["category"] for i in item_info])
                    
                    if category in event_categories:
                        category_low_count += 1
            
            # If rated low 2+ times, add to disliked
            if category_low_count >= 2 and category not in prefs["disliked_categories"]:
                prefs["disliked_categories"].append(category)
    
    # Save updated preferences
    supabase.table("user_preferences").update(prefs).eq("user_id", user_id).execute()
    
    return prefs


def get_outfit_recommendations_with_learning(user_id: str):
    """
    Analyze past ratings to improve future recommendations.
    
    Returns insights about what the user likes/dislikes.
    """
    # Fetch all ratings
    ratings = (
        supabase.table("events")
        .select("*")
        .eq("user_id", user_id)
        .eq("event_name", "rate_outfit")
        .execute()
        .data
    )
    
    if not ratings:
        return {
            "total_ratings": 0,
            "avg_rating": 0,
            "insights": "No ratings yet - rate some outfits to help us learn your style!"
        }
    
    # Calculate statistics
    total_ratings = len(ratings)
    total_score = sum(r.get("properties", {}).get("rating", 0) for r in ratings)
    avg_rating = total_score / total_ratings if total_ratings > 0 else 0
    
    # Find highly rated outfit patterns
    high_rated = [r for r in ratings if r.get("properties", {}).get("rating", 0) >= 4]
    low_rated = [r for r in ratings if r.get("properties", {}).get("rating", 0) <= 2]
    
    # Analyze which types of outfits are liked
    liked_outfit_types = {}
    for rating in high_rated:
        outfit_type = rating.get("properties", {}).get("outfit_type", "unknown")
        liked_outfit_types[outfit_type] = liked_outfit_types.get(outfit_type, 0) + 1
    
    insights = f"You've rated {total_ratings} outfits with an average of {avg_rating:.1f} stars. "
    
    if high_rated:
        insights += f"You love {len(high_rated)} outfits! "
        most_liked_type = max(liked_outfit_types.items(), key=lambda x: x[1])[0] if liked_outfit_types else "N/A"
        insights += f"Your favorite outfit type seems to be: {most_liked_type}. "
    
    if low_rated:
        insights += f"You've disliked {len(low_rated)} outfits - we'll avoid similar combinations."
    
    return {
        "total_ratings": total_ratings,
        "avg_rating": round(avg_rating, 2),
        "high_rated_count": len(high_rated),
        "low_rated_count": len(low_rated),
        "insights": insights,
        "favorite_outfit_types": liked_outfit_types
    }



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
        outfits=formatted_outfits,
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
async def rate_outfit(request: RateOutfitRequest):
    """
    Let users rate outfit suggestions to improve future recommendations.
    
    This is where the AI learns your style preferences!
    - High ratings (4-5): We learn what styles you love
    - Low ratings (1-2): We learn what to avoid
    
    Returns:
        Confirmation with updated preferences
    """
    # Validate rating
    if request.rating < 1 or request.rating > 5:
        return {"error": "Rating must be between 1 and 5 stars"}
    
    # Track the rating event
    track_event(
        request.user_id,
        "rate_outfit",
        {
            "outfit_number": request.outfit_number,
            "rating": request.rating,
            "items": request.items,
            "outfit_type": request.outfit_type
        }
    )
    
    # Update user preferences based on this rating
    updated_prefs = update_preferences_from_rating(request.user_id, request.rating, request.items)
    
    # Get insights from all ratings
    insights = get_outfit_recommendations_with_learning(request.user_id)
    
    response_message = ""
    if request.rating >= 4:
        response_message = f"Great! We'll recommend more {request.outfit_type} outfits like this one. "
    elif request.rating <= 2:
        response_message = f"Got it! We'll avoid similar combinations in the future. "
    else:
        response_message = "Thanks for your feedback! "
    
    return {
        "status": "rating recorded",
        "message": response_message,
        "outfit_number": request.outfit_number,
        "rating": request.rating,
        "updated_preferences": {
            "preferred_styles": updated_prefs.get("preferred_styles", []),
            "disliked_categories": updated_prefs.get("disliked_categories", []),
            "formality_score": updated_prefs.get("formality_score", 0.5)
        },
        "learning_insights": insights
    }

@app.get("/my_style_profile/{user_id}")
async def get_style_profile(user_id: str = "test_user"):
    """
    Get a complete view of what the AI has learned about your style.
    
    Shows:
    - Your preferred styles
    - Categories you dislike
    - Formality preference
    - Rating history and patterns
    """
    # Get preferences
    prefs = supabase.table("user_preferences").select("*").eq("user_id", user_id).execute().data
    
    if not prefs:
        return {
            "error": "No profile found",
            "message": "Start uploading items and rating outfits to build your style profile!"
        }
    
    profile = prefs[0]
    
    # Get rating insights
    insights = get_outfit_recommendations_with_learning(user_id)
    
    # Get wardrobe summary
    items = supabase.table("wardrobe_items").select("*").eq("user_id", user_id).execute().data
    
    category_counts = {}
    style_counts = {}
    
    for item in items:
        cat = item.get("category", "unknown")
        style = item.get("style", "unknown")
        
        category_counts[cat] = category_counts.get(cat, 0) + 1
        if style != "unknown":
            style_counts[style] = style_counts.get(style, 0) + 1
    
    return {
        "user_id": user_id,
        "style_preferences": {
            "preferred_styles": profile.get("preferred_styles", []),
            "disliked_categories": profile.get("disliked_categories", []),
            "formality_score": profile.get("formality_score", 0.5),
            "formality_label": (
                "Very Casual" if profile.get("formality_score", 0.5) < 0.3 else
                "Casual" if profile.get("formality_score", 0.5) < 0.5 else
                "Smart Casual" if profile.get("formality_score", 0.5) < 0.7 else
                "Formal"
            )
        },
        "wardrobe_stats": {
            "total_items": len(items),
            "categories": category_counts,
            "styles": style_counts
        },
        "rating_history": insights,
        "recommendation_tip": (
            f"Based on your {insights['total_ratings']} ratings, "
            f"we recommend outfits with a formality level of {profile.get('formality_score', 0.5):.1f}. "
            f"{insights['insights']}"
        )
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