import requests
import os

BACKBOARD_API_KEY = os.getenv("BACKBOARD_API_KEY")
BACKBOARD_URL = "https://api.backboard.ai/v1/chat/completions"

def call_backboard(model: str, system_prompt: str, user_prompt: str, memory=None):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "memory": memory or []
    }

    headers = {
        "Authorization": f"Bearer {BACKBOARD_API_KEY}",
        "Content-Type": "application/json"
    }

    resp = requests.post(BACKBOARD_URL, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

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
        memory=events
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