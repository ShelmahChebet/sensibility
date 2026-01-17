import requests
import os
import json
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from dotenv import load_dotenv
load_dotenv()

_ASSISTANTS = {}  # model_name -> assistant_id

# Define device for BLIP/torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize globally
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

BACKBOARD_API_KEY = os.getenv("BACKBOARD_API_KEY")
BASE_URL = "https://app.backboard.io/api"
HEADERS = {"X-API-Key": BACKBOARD_API_KEY}


def get_or_create_assistant_for_model(model_name: str):
    """
    Returns the assistant_id for a given model. Creates assistant if needed.
    """
    if model_name in _ASSISTANTS:
        return _ASSISTANTS[model_name]

    print(f"Creating Backboard assistant for model {model_name}...")
    resp = requests.post(
        f"{BASE_URL}/assistants",
        json={
            "name": f"Sensibility Stylist ({model_name})",
            "system_prompt": (
                "You are a fashion assistant. "
                "Given a detailed description of a clothing item, return a concise, natural style label "
                "that reflects when and where the item would be worn."
            ),
            "model": model_name
        },
        headers=HEADERS
    )
    resp.raise_for_status()
    assistant_id = resp.json()["assistant_id"]
    _ASSISTANTS[model_name] = assistant_id
    return assistant_id


def create_thread(assistant_id: str):
    resp = requests.post(
        f"{BASE_URL}/assistants/{assistant_id}/threads",
        headers=HEADERS,
        json={}
    )
    resp.raise_for_status()
    return resp.json()["thread_id"]


def send_message(thread_id: str, content: str):
    """
    Send a message to a Backboard thread.
    Using data= instead of json= as per Backboard docs.
    """
    payload = {
        "content": content,
        "stream": "false",
        "memory": "Auto"
    }
    
    print(f"\n=== SENDING TO BACKBOARD ===")
    print(f"Thread ID: {thread_id}")
    print(f"Payload: {payload}")
    
    resp = requests.post(
        f"{BASE_URL}/threads/{thread_id}/messages",
        headers=HEADERS,
        data=payload  # Changed from json= to data=
    )
    
    print(f"Status Code: {resp.status_code}")
    
    if resp.status_code != 200:
        print(f"ERROR Response Body: {resp.text}")
        try:
            error_json = resp.json()
            print(f"ERROR JSON: {json.dumps(error_json, indent=2)}")
        except:
            pass
    
    resp.raise_for_status()
    data = resp.json()
    print(f"Success Response: {json.dumps(data, indent=2)}")
    print("=== END BACKBOARD CALL ===\n")

    try:
        if "messages" in data and len(data["messages"]) > 0:
            return data["messages"][0]["content"].strip()
        elif "content" in data:
            return data["content"].strip()
        else:
            return ""
    except Exception as e:
        print(f"Error parsing response: {e}")
        return ""


def caption_image(image: Image.Image) -> str:
    inputs = blip_processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = blip_model.generate(
            **inputs,
            max_new_tokens=30
        )

    caption = blip_processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption.lower().strip()


def infer_item_style(image: Image.Image) -> str:
    caption = caption_image(image)
    assistant_id = get_or_create_assistant_for_model("openai/gpt-4o-mini")
    thread_id = create_thread(assistant_id)

    prompt = (
        f"Item description: {caption}\n"
        "Return a concise style label describing the type of outfit or occasion this item is best for. "
        "Be very specific, e.g.: 'office blouse', 'going out top', 'beach dress', 'casual joggers'. "
        "Return ONLY the label, no extra words or punctuation."
    )

    response = send_message(thread_id, prompt)
    style = response.strip().lower()
    if not style or len(style) > 50:
        return "unknown"
    return style


def call_backboard(model: str, system_prompt: str, user_prompt: str) -> str:
    """
    Simplified - combine system and user prompts into content.
    """
    assistant_id = get_or_create_assistant_for_model(model)
    thread_id = create_thread(assistant_id)
    
    # Combine prompts
    content = f"{system_prompt}\n\n{user_prompt}"
    
    return send_message(thread_id, content)


def analyze_user_style(events, wardrobe_summary):
    system_prompt = """
You are a fashion behavior analyst.
Your job is to infer user preferences over time from behavior logs.
Output a concise JSON summary with this exact structure:
{
  "preferred_styles": ["style1", "style2"],
  "disliked_categories": ["category1"],
  "formality_score": 0.5
}
DO NOT wrap the JSON in markdown code blocks or backticks.
"""
    
    # Format events as a readable string
    events_str = json.dumps(events, indent=2) if events else "No events yet"
    
    user_prompt = f"""
User wardrobe summary:
{wardrobe_summary}

Behavior events:
{events_str}

Based on this information, infer:
- preferred aesthetics (list of strings)
- formality preference (float between 0 and 1, where 0 is very casual and 1 is very formal)
- disliked categories (list of strings)

Return ONLY valid JSON with the structure specified above.
DO NOT use markdown code blocks or backticks.
"""

    raw = call_backboard(
        model="anthropic/claude-3-haiku",
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    # Strip markdown code blocks if present
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]  # Remove ```json
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]  # Remove ```
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]  # Remove trailing ```
    cleaned = cleaned.strip()

    try:
        structured = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from LLM: {e}")
        print(f"Raw response was: {raw}")
        structured = {
            "preferred_styles": [], 
            "disliked_categories": [], 
            "formality_score": 0.5
        }

    structured.setdefault("preferred_styles", [])
    structured.setdefault("disliked_categories", [])
    structured.setdefault("formality_score", 0.5)

    return structured


def explain_recommendations(prompt, outfits, user_profile):
    """
    Generate AI explanation for complete outfit recommendations.
    
    Args:
        prompt: User's original request
        outfits: List of complete outfit combinations
        user_profile: User's style preferences
    
    Returns:
        Dict with explanation, top_reason, and styling_tips
    """
    system_prompt = """
You are a professional fashion styling assistant.
Explain complete outfit recommendations clearly and helpfully.
Return your response strictly as valid JSON with this structure:
{
  "explanation": "detailed explanation of why these complete outfits work well together and match the user's request",
  "top_reason": "the main reason these specific outfit combinations were chosen",
  "styling_tips": "practical tips on how to wear and accessorize each outfit"
}
DO NOT wrap the JSON in markdown code blocks or backticks.
"""
    
    outfits_str = json.dumps(outfits, indent=2) if outfits else "No outfits"
    profile_str = json.dumps(user_profile, indent=2) if user_profile else "No profile"
    
    user_prompt = f"""
User asked for: "{prompt}"

User style profile:
{profile_str}

Complete outfit combinations suggested:
{outfits_str}

Provide a comprehensive styling explanation that covers:
1. Why these complete outfit combinations were chosen (considering the items work well together)
2. How each outfit matches their request and personal style preferences
3. Specific styling tips for wearing these outfits (accessories, shoes, occasions)

Return ONLY valid JSON with the exact structure specified above.
DO NOT use markdown code blocks or backticks.
"""

    raw = call_backboard(
        model="openai/gpt-4o-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    # Strip markdown code blocks if present
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]  # Remove ```json
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]  # Remove ```
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]  # Remove trailing ```
    cleaned = cleaned.strip()

    try:
        structured = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from LLM: {e}")
        print(f"Raw response was: {raw}")
        structured = {
            "explanation": "Sorry, could not generate explanation.",
            "top_reason": "",
            "styling_tips": ""
        }

    structured.setdefault("explanation", "Sorry, could not generate explanation.")
    structured.setdefault("top_reason", "")
    structured.setdefault("styling_tips", "")

    return structured