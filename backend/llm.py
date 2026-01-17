import requests
import os
import json
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from dotenv import load_dotenv
load_dotenv()

_ASSISTANT_ID = None


# Define device for BLIP/torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize globally
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

BACKBOARD_API_KEY = os.getenv("BACKBOARD_API_KEY")
BASE_URL = "https://app.backboard.io/api"
HEADERS = {"X-API-Key": BACKBOARD_API_KEY}

def create_assistant():
    resp = requests.post(
        f"{BASE_URL}/assistants",
        json={
            "name": "Sensibility Stylist",
            "system_prompt": (
                "You are a fashion assistant. "
                "Given a detailed description of a clothing item, return a concise, natural style label that reflects when and where the item would be worn."
            ),
        },
        headers=HEADERS,
    )
    resp.raise_for_status()
    return resp.json()["assistant_id"]

def get_or_create_assistant():
    global _ASSISTANT_ID
    if _ASSISTANT_ID is None:
        print("Creating Backboard assistant...")
        _ASSISTANT_ID = create_assistant()
    return _ASSISTANT_ID

def create_thread(assistant_id: str):
    resp = requests.post(
        f"{BASE_URL}/assistants/{assistant_id}/threads",
        json={},
        headers=HEADERS,
    )
    resp.raise_for_status()
    return resp.json()["thread_id"]

def send_message(thread_id: str, content: str):
    resp = requests.post(
        f"{BASE_URL}/threads/{thread_id}/messages",
        headers=HEADERS,
        data={
            "content": content,
            "stream": "false",
            "memory": "Auto",
        },
    )
    resp.raise_for_status()
    return resp.json().get("content")

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
    assistant_id = get_or_create_assistant()
    thread_id = create_thread(assistant_id)

    prompt = (
        f"Item description: {caption}\n"
        "Return a concise style label describing the type of outfit or occasion this item is best for. "
        "Be very specific, e.g.: 'office blouse', 'going out top', 'beach dress', 'casual joggers'. "
        "Return ONLY the label, no extra words or punctuation."
    )

    response = send_message(thread_id, prompt)

    style = response.strip().lower()

    # ultra-light safety only
    if not style or len(style) > 50:
        return "unknown"

    return style

def call_backboard(model: str, system_prompt: str, user_prompt: str, memory=None):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "memory": memory or []
    }

    headers = {"X-API-Key": BACKBOARD_API_KEY}
    resp = requests.post(f"{BASE_URL}/assistants/{get_or_create_assistant()}/threads", json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    print("Backboard raw response:", data)
    
    # fallback
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return json.dumps(data)

def analyze_user_style(events, wardrobe_summary):
    system_prompt = """
    You are a fashion behavior analyst.
    Your job is to infer user preferences over time from behavior logs.
    Output a concise JSON summary.
    """

    user_prompt = f"""
    User wardrobe summary:
    {wardrobe_summary}

    Behavior events:
    {events}

    Infer:
    - preferred aesthetics
    - formality preference (0–1)
    - disliked categories
    """

    raw = call_backboard(
        model="anthropic/claude-3-haiku",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        memory=[json.loads(json.dumps(e)) for e in events]
    )
    # Attempt to parse JSON safely
    try:
        structured = json.loads(raw)
    except json.JSONDecodeError:
        structured = {"preferred_styles": [], "disliked_categories": [], "formality_score": 0.5}

    # Ensure all keys exist
    structured.setdefault("preferred_styles", [])
    structured.setdefault("disliked_categories", [])
    structured.setdefault("formality_score", 0.5)

    return structured

def explain_recommendations(prompt, matches, user_profile):
    system_prompt = """
    You are a fashion assistant.
    Explain recommendations clearly and helpfully.
    """

    user_prompt = f"""
    User asked for: "{prompt}"

    User style profile:
    {user_profile}

    Top matches:
    {matches}

    Explain why these items were recommended and how to style them.
    """
    raw = call_backboard(
        model="openai/gpt-4o-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    # Attempt to parse JSON safely
    try:
        structured = json.loads(raw)
    except json.JSONDecodeError:
        structured = {
            "explanation": "Sorry, could not generate explanation.",
            "top_reason": ""
        }

    # Ensure keys exist
    structured.setdefault("explanation", "Sorry, could not generate explanation.")
    structured.setdefault("top_reason", "")

    return structured

