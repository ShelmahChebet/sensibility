"""
Outfit Curation Logic
Combines individual clothing items into complete, coordinated outfits
"""

import torch
import torch.nn.functional as F
from itertools import combinations
import random

def calculate_outfit_compatibility(items, clip_model, clip_processor, device):
    """
    Calculate how well items go together based on their visual embeddings.
    
    Args:
        items: List of wardrobe items with embeddings
        clip_model, clip_processor, device: CLIP model components
    
    Returns:
        Compatibility score (0-100)
    """
    if len(items) < 2:
        return 0
    
    # Calculate pairwise similarity between all items in the outfit
    embeddings = [torch.tensor(item["embedding"]) for item in items]
    total_similarity = 0
    count = 0
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            emb1 = embeddings[i] / embeddings[i].norm(p=2)
            emb2 = embeddings[j] / embeddings[j].norm(p=2)
            similarity = F.cosine_similarity(emb1, emb2, dim=0)
            total_similarity += similarity
            count += 1
    
    if count == 0:
        return 0
    
    avg_similarity = total_similarity / count
    # Convert to 0-100 scale
    return float((avg_similarity + 1) / 2 * 100)


def categorize_items(items):
    """
    Group items by category for outfit building.
    
    Returns:
        Dictionary with categories as keys and items as values
    """
    categories = {
        "tops": [],
        "bottoms": [],
        "dresses": [],
        "outerwear": [],
        "shoes": [],
        "accessories": []
    }
    
    category_mapping = {
        "shirt": "tops",
        "blouse": "tops",
        "top": "tops",
        "sweater": "tops",
        "hoodie": "tops",
        "t-shirt": "tops",
        "tank": "tops",
        
        "pants": "bottoms",
        "jeans": "bottoms",
        "skirt": "bottoms",
        "shorts": "bottoms",
        "trousers": "bottoms",
        
        "dress": "dresses",
        
        "jacket": "outerwear",
        "coat": "outerwear",
        "blazer": "outerwear",
        "cardigan": "outerwear",
        
        "shoes": "shoes",
        "sneakers": "shoes",
        "boots": "shoes",
        "heels": "shoes",
        
        "bag": "accessories",
        "hat": "accessories",
        "scarf": "accessories",
        "belt": "accessories",
        "jewelry": "accessories"
    }
    
    for item in items:
        category = item.get("category", "").lower()
        mapped_category = category_mapping.get(category, "accessories")
        categories[mapped_category].append(item)
    
    return categories


def score_outfit_for_prompt(outfit_items, text_embedding, formality_preference=0.5):
    """
    Score an outfit based on how well it matches the user's prompt and preferences.
    
    Args:
        outfit_items: List of items in the outfit
        text_embedding: CLIP embedding of the user's text prompt
        formality_preference: User's formality score (0-1)
    
    Returns:
        Total outfit score
    """
    scores = []
    
    for item in outfit_items:
        item_embedding = torch.tensor(item["embedding"])
        item_embedding = item_embedding / item_embedding.norm(p=2)
        similarity = F.cosine_similarity(text_embedding, item_embedding, dim=0)
        base_score = float((similarity + 1) / 2 * 100)
        
        # Adjust for formality
        item_formality = item.get("formality", 0.5)
        formality_match = 1 - abs(item_formality - formality_preference)
        
        adjusted_score = base_score * (0.7 + 0.3 * formality_match)
        scores.append(adjusted_score)
    
    return sum(scores) / len(scores) if scores else 0


def generate_outfits(
    items,
    text_embedding,
    user_preferences,
    clip_model,
    clip_processor,
    device,
    num_outfits=3
):
    """
    Generate complete outfit suggestions.
    
    Args:
        items: All wardrobe items
        text_embedding: CLIP embedding of user's prompt
        user_preferences: User style preferences
        clip_model, clip_processor, device: CLIP components
        num_outfits: Number of outfit suggestions to generate
    
    Returns:
        List of outfit dictionaries
    """
    categorized = categorize_items(items)
    outfits = []
    
    formality_pref = user_preferences.get("formality_score", 0.5)
    
    # Strategy 1: Dress-based outfits
    if categorized["dresses"]:
        for dress in categorized["dresses"][:3]:  # Try top 3 dresses
            outfit = {
                "items": [dress],
                "type": "dress"
            }
            
            # Add outerwear if available and weather-appropriate
            if categorized["outerwear"]:
                outfit["items"].append(categorized["outerwear"][0])
            
            # Add shoes if available
            if categorized["shoes"]:
                outfit["items"].append(categorized["shoes"][0])
            
            # Calculate scores
            compatibility = calculate_outfit_compatibility(
                outfit["items"], clip_model, clip_processor, device
            )
            prompt_match = score_outfit_for_prompt(
                outfit["items"], text_embedding, formality_pref
            )
            
            outfit["compatibility_score"] = compatibility
            outfit["prompt_match_score"] = prompt_match
            outfit["overall_score"] = (compatibility * 0.4 + prompt_match * 0.6)
            
            outfits.append(outfit)
    
    # Strategy 2: Top + Bottom combinations
    if categorized["tops"] and categorized["bottoms"]:
        # Generate multiple combinations
        top_bottom_combos = []
        
        for top in categorized["tops"][:4]:  # Top 4 tops
            for bottom in categorized["bottoms"][:4]:  # Top 4 bottoms
                outfit = {
                    "items": [top, bottom],
                    "type": "separates"
                }
                
                # Add outerwear if beneficial
                if categorized["outerwear"]:
                    outfit["items"].append(categorized["outerwear"][0])
                
                # Add shoes if available
                if categorized["shoes"]:
                    outfit["items"].append(categorized["shoes"][0])
                
                # Calculate scores
                compatibility = calculate_outfit_compatibility(
                    outfit["items"], clip_model, clip_processor, device
                )
                prompt_match = score_outfit_for_prompt(
                    outfit["items"], text_embedding, formality_pref
                )
                
                outfit["compatibility_score"] = compatibility
                outfit["prompt_match_score"] = prompt_match
                outfit["overall_score"] = (compatibility * 0.4 + prompt_match * 0.6)
                
                top_bottom_combos.append(outfit)
        
        # Add best combinations
        top_bottom_combos.sort(key=lambda x: x["overall_score"], reverse=True)
        outfits.extend(top_bottom_combos[:5])  # Take top 5 combinations
    
    # Sort all outfits by overall score
    outfits.sort(key=lambda x: x["overall_score"], reverse=True)
    
    # Take top N unique outfits
    final_outfits = []
    seen_combinations = set()
    
    for outfit in outfits:
        # Create a signature for this outfit
        item_ids = tuple(sorted([item["filename"] for item in outfit["items"]]))
        
        if item_ids not in seen_combinations:
            seen_combinations.add(item_ids)
            final_outfits.append(outfit)
            
            if len(final_outfits) >= num_outfits:
                break
    
    return final_outfits


def format_outfit_response(outfits):
    """
    Format outfit data for API response.
    
    Returns:
        List of formatted outfit dictionaries
    """
    formatted = []
    
    for i, outfit in enumerate(outfits, 1):
        formatted_outfit = {
            "outfit_number": i,
            "type": outfit["type"],
            "items": [
                {
                    "filename": item["filename"],
                    "category": item["category"],
                    "style": item.get("style", "unknown"),
                    "image_url": item.get("image_url", "")
                }
                for item in outfit["items"]
            ],
            "scores": {
                "compatibility": round(outfit["compatibility_score"], 2),
                "prompt_match": round(outfit["prompt_match_score"], 2),
                "overall": round(outfit["overall_score"], 2)
            }
        }
        formatted.append(formatted_outfit)
    
    return formatted