# Bakery Customer Feedback — Week 2 ML Business Applications

**Business question:** Which bakery products generate positive or negative customer feedback, and which comments need manual review?

## Labels (exactly one for each written comment)

- **positive**: overall favorable feedback, even with a minor complaint.
- **negative**: overall unfavorable feedback, even with minor praise.
- **mixed**: meaningful positive and negative feedback, neither clearly dominant.
- **unclear**: technical fallback when the model returns an invalid or unparseable label; needs manual review.

**Product identification is a separate field**, not a second sentiment label. Groq receives the exact 37-item catalog from the Week 1 bakery dataset. It is prompted to ignore case, handle spelling mistakes, plurals, accented names, and reasonable synonyms, while avoiding guesses between similar products. If no specific product can be confidently identified, it returns `General`.

## Customer side

Leave a rating (1–5 stars), written comment, **or both**. At least one is required. Written comments go to Groq **once** for a sentiment and product. Rating-only submissions are saved without sentiment; product is `General`. Saved columns: `timestamp, comment, rating, label, product`.

## Owner side

- Default **General Overview**, with dropdown for individual product (and `General`) drill-down.
- Three headline metrics: feedback received, average rating (among submissions with stars), and manual-review count.
- Two charts: stacked sentiment by product; highest-rated five products. **Only products with at least three star ratings qualify for the highest-rated chart.** This threshold is displayed above the chart.
- Top five most-reviewed products, as a table ranked by number of **written comments** (not star-only submissions).
- Manual-review queue shows full comments: mixed/unclear, positive with 1–2 stars, or negative with 4–5 stars.
- Product-specific counts, sentiment breakdown, and comments in dropdown drill-down.

## Files

- `app.py` — Streamlit customer and owner pages
- `labeler.py` — single Groq request and strict result validation
- `bakery_products.csv` — exact 37 products extracted from Week 1
- `demo_comments.csv` — optional sample comments for manual testing (not automatically loaded)
- `requirements.txt` — dependencies

## Run locally

```bash
pip install -r requirements.txt
mkdir -p .streamlit
```

In `.streamlit/secrets.toml` (never commit this file), write:

```toml
GROQ_API_KEY = "your-groq-key-here"
```

Then run:

```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push `app.py`, `labeler.py`, `bakery_products.csv`, `requirements.txt`, and this README to a **public GitHub repository**. Never include your API key or `.streamlit/secrets.toml`.
2. Create an app at https://share.streamlit.io using `app.py` as the main file.
3. In the app's **Secrets** setting, add `GROQ_API_KEY = "your-groq-key-here"`.
4. Open the customer and owner tabs and test them. The Groq model is configured in `labeler.py`; see https://console.groq.com/docs/models if that model is unavailable.

**Storage limitation:** The app saves feedback in `feedback.csv` on its local filesystem. On free Streamlit Cloud, this may be deleted when the app restarts/redeploys and multiple app instances can cause inconsistencies. Download CSV feedback regularly. A production version should use persistent storage.

**Data limitations:** The five-star rating from a rating-only entry cannot be assigned to any specific product; it is saved as `General`. The sentiment chart only uses written comments; the rating chart only uses entries containing stars. The model may still misinterpret ambiguous comments despite spelling-tolerant instructions.
