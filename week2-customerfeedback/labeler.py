"""One Groq request returns one sentiment label and one menu product."""
import csv
import json
from pathlib import Path

import requests
import streamlit as st

LABELS = ("positive", "negative", "mixed")
MODEL = "openai/gpt-oss-20b"  # If retired, select a current model at console.groq.com/docs/models
URL = "https://api.groq.com/openai/v1/chat/completions"
CATALOG = Path(__file__).with_name("bakery_products.csv")


def load_products():
    with CATALOG.open(encoding="utf-8-sig", newline="") as file:
        return [row["product"] for row in csv.DictReader(file)]


def parse_classification(answer, products):
    """Validate the model output rather than accepting invented product names."""
    try:
        # Models occasionally wrap JSON in markdown fences.
        cleaned = answer.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(cleaned)
        label = str(result.get("label", "")).strip().lower()
        product = str(result.get("product", "")).strip()
    except (ValueError, AttributeError, TypeError):
        return "unclear", "General"

    label = label if label in LABELS else "unclear"
    lookup = {name.casefold(): name for name in products}
    # Product identification happens in the LLM. Strict validation prevents hallucinated names.
    product = lookup.get(product.casefold(), "General")
    return label, product


def label_comment(comment):
    """Return (sentiment, product) for a nonempty customer comment."""
    products = load_products()
    key = st.secrets.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing from Streamlit secrets.")

    menu = "\n".join(f"- {p}" for p in products)
    instructions = f"""You categorize customer comments for a bakery.
Return exactly ONE JSON object, with exactly the string keys "label" and "product".
Do not include markdown or explanatory text.

Allowed sentiment labels:
- positive: Overall favorable feedback, including praise with a minor complaint.
- negative: Overall unfavorable feedback, including complaints with minor praise.
- mixed: Both meaningful positive and negative feedback, with neither clearly dominating.
Choose exactly ONE sentiment label. Interpret sarcasm appropriately.

Identify the MAIN bakery product mentioned using ONLY the approved catalog below.
- Ignore uppercase/lowercase, accents, punctuation, and singular/plural differences.
- Recognize ordinary spelling mistakes, typos, aliases, and natural-language translations
  if the intended product is reasonably clear.
- Be careful with similar items: a baguette is not necessarily a DEMI BAGUETTE,
  CEREAL BAGUETTE, or TRADITIONAL BAGUETTE; a croissant is not necessarily
  a CROISSANT AMANDES.
- If there is no identifiable specific product, or several products are discussed
  without a clear main subject, choose "General". Do not guess.
- Output the EXACT catalog spelling for any identified product.

Approved bakery product catalog:
{menu}

Example output: {{"label":"mixed","product":"BAGUETTE"}}"""

    response = requests.post(
        URL,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": MODEL,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": comment},
            ],
        },
        timeout=30,
    )
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"]
    return parse_classification(answer, products)
