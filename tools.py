"""
tools.py

The three required FitFindr tools. Each tool is a standalone function that
can be called and tested independently before being wired into the agent loop.

Complete and test each tool before moving to agent.py.

Tools:
    search_listings(description, size, max_price)  → list[dict]
    suggest_outfit(new_item, wardrobe)              → str
    create_fit_card(outfit, new_item)               → str
"""

import os

from dotenv import load_dotenv
from groq import Groq

from utils.data_loader import load_listings

load_dotenv()


# ── Groq client ───────────────────────────────────────────────────────────────

def _get_groq_client():
    """Initialize and return a Groq client using GROQ_API_KEY from .env."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to a .env file in the project root."
        )
    return Groq(api_key=api_key)


# ── Tool 1: search_listings ───────────────────────────────────────────────────

def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    """
    Search the mock listings dataset for items matching the description,
    optional size, and optional price ceiling.

    Args:
        description: Keywords describing what the user is looking for
                     (e.g., "vintage graphic tee").
        size:        Size string to filter by, or None to skip size filtering.
                     Matching is case-insensitive (e.g., "M" matches "S/M").
        max_price:   Maximum price (inclusive), or None to skip price filtering.

    Returns:
        A list of matching listing dicts, sorted by relevance (best match first).
        Returns an empty list if nothing matches — does NOT raise an exception.

    Each listing dict has the following fields:
        id, title, description, category, style_tags (list), size,
        condition, price (float), colors (list), brand, platform

    TODO:
        1. Load all listings with load_listings().
        2. Filter by max_price and size (if provided).
        3. Score each remaining listing by keyword overlap with `description`.
        4. Drop any listings with a score of 0 (no relevant matches).
        5. Sort by score, highest first, and return the listing dicts.

    Before writing code, fill in the Tool 1 section of planning.md.
    """
    # Load all listings
    listings = load_listings()
    
    # Filter by max_price if provided
    if max_price is not None:
        listings = [lst for lst in listings if lst["price"] <= max_price]
    
    # Filter by size if provided (case-insensitive, partial match)
    if size is not None:
        size_lower = size.lower()
        listings = [
            lst for lst in listings
            if size_lower in lst["size"].lower()
        ]
    
    # Score each listing by keyword overlap
    description_words = set(description.lower().split())
    scored_listings = []
    
    for listing in listings:
        # Combine searchable fields
        searchable_text = (
            listing["title"].lower() + " " +
            listing["description"].lower() + " " +
            " ".join(listing["style_tags"]).lower()
        )
        
        # Count keyword matches
        searchable_words = set(searchable_text.split())
        score = len(description_words & searchable_words)
        
        # Keep only listings with score > 0
        if score > 0:
            scored_listings.append((listing, score))
    
    # Sort by score descending and return listing dicts only
    scored_listings.sort(key=lambda x: x[1], reverse=True)
    return [listing for listing, score in scored_listings]


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────

def suggest_outfit(new_item: dict, wardrobe: dict) -> str:
    """
    Given a thrifted item and the user's wardrobe, suggest 1–2 complete outfits.

    Args:
        new_item: A listing dict (the item the user is considering buying).
        wardrobe: A wardrobe dict with an 'items' key containing a list of
                  wardrobe item dicts. May be empty — handle this gracefully.

    Returns:
        A non-empty string with outfit suggestions.
        If the wardrobe is empty, offer general styling advice for the item
        rather than raising an exception or returning an empty string.

    TODO:
        1. Check whether wardrobe['items'] is empty.
        2. If empty: call the LLM with a prompt for general styling ideas
           (what kinds of items pair well, what vibe it suits, etc.).
        3. If not empty: format the wardrobe items into a prompt and ask
           the LLM to suggest specific outfit combinations using the new item
           and named pieces from the wardrobe.
        4. Return the LLM's response as a string.

    Before writing code, fill in the Tool 2 section of planning.md.
    """
    try:
        client = _get_groq_client()
        
        # Check if wardrobe items list is empty
        wardrobe_items = wardrobe.get("items", [])
        
        if not wardrobe_items:
            # Generate general styling advice
            prompt = f"""
You are a fashion styling assistant. A user is considering buying this thrifted item:

Title: {new_item.get('title', 'Unknown')}
Description: {new_item.get('description', '')}
Category: {new_item.get('category', 'Unknown')}
Style Tags: {', '.join(new_item.get('style_tags', []))}
Size: {new_item.get('size', 'Unknown')}
Colors: {', '.join(new_item.get('colors', []))}
Price: ${new_item.get('price', 0):.2f}

Their wardrobe is currently empty. Provide general styling advice for this item. Explain:
- What types of pants, shoes, layers, and accessories pair well with it
- The vibe and occasions this item suits
- General styling tips to make this item work in multiple outfits

Keep the response warm, encouraging, and helpful.
"""
        else:
            # Format wardrobe items for the prompt
            wardrobe_text = "\n".join([
                f"- {item.get('name', 'Item')}: {item.get('description', '')} ({item.get('category', 'Unknown')})"
                for item in wardrobe_items
            ])
            
            prompt = f"""
You are a fashion styling assistant. A user is considering buying this thrifted item:

Title: {new_item.get('title', 'Unknown')}
Description: {new_item.get('description', '')}
Category: {new_item.get('category', 'Unknown')}
Style Tags: {', '.join(new_item.get('style_tags', []))}
Size: {new_item.get('size', 'Unknown')}
Colors: {', '.join(new_item.get('colors', []))}
Price: ${new_item.get('price', 0):.2f}

Their current wardrobe contains:
{wardrobe_text}

Suggest 1–2 complete outfits using this new item combined with pieces from their existing wardrobe.
Mention specific pieces by name from their wardrobe. Be specific and creative.
Keep the response warm, enthusiastic, and practical.
"""
        
        message = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        
        return message.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Sorry, I couldn't generate outfit suggestions right now. Error: {str(e)}"


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def create_fit_card(outfit: str, new_item: dict) -> str:
    """
    Generate a short, shareable outfit caption for the thrifted find.

    Args:
        outfit:   The outfit suggestion string from suggest_outfit().
        new_item: The listing dict for the thrifted item.

    Returns:
        A 2–4 sentence string usable as an Instagram/TikTok caption.
        If outfit is empty or missing, return a descriptive error message
        string — do NOT raise an exception.

    The caption should:
    - Feel casual and authentic (like a real OOTD post, not a product description)
    - Mention the item name, price, and platform naturally (once each)
    - Capture the outfit vibe in specific terms
    - Sound different each time for different inputs (use higher LLM temperature)

    TODO:
        1. Guard against an empty or whitespace-only outfit string.
        2. Build a prompt that gives the LLM the item details and the outfit,
           and asks for a caption matching the style guidelines above.
        3. Call the LLM and return the response.

    Before writing code, fill in the Tool 3 section of planning.md.
    """
    # Guard against empty or whitespace-only outfit
    if not outfit or not outfit.strip():
        return "Unable to generate fit card because outfit information is missing."
    
    try:
        client = _get_groq_client()
        
        item_title = new_item.get('title', 'thrifted piece')
        item_price = new_item.get('price', 0)
        item_platform = new_item.get('platform', 'thrift')
        
        prompt = f"""
You are a fashion social media expert. Generate a casual, authentic Instagram/TikTok style OOTD (Outfit of the Day) caption.

Item Details:
- Title: {item_title}
- Price: ${item_price:.2f}
- Platform: {item_platform}
- Category: {new_item.get('category', 'unknown')}
- Style Tags: {', '.join(new_item.get('style_tags', []))}

Outfit Suggestion:
{outfit}

Write a 2–4 sentence caption that:
- Feels casual and authentic (like a real person sharing their outfit, not a product description)
- Mentions the item title, price, and platform naturally, once each
- Captures the outfit vibe in specific, vivid terms
- Sounds conversational and engaging

Caption:
"""
        
        message = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
        )
        
        return message.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Sorry, I couldn't generate a fit card right now. Error: {str(e)}"
