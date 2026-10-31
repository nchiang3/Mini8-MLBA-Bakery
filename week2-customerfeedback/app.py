"""Bakery customer feedback: Week 2 ML Business Applications."""
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from labeler import label_comment, load_products

# ---- Settings (same structure as the professor's pizza starter) ---------
TITLE = "Bakery Customer Feedback"
DATA_FILE = Path(__file__).with_name("feedback.csv")
DEMO_FILE = Path(__file__).with_name("bakery_sample_feedback_35.csv")
LOW_RATING, HIGH_RATING = 2, 4
COLORS = {"positive": "#2BB38A", "mixed": "#E6AD47", "negative": "#D35F58", "unclear": "#999999"}
LABEL_ORDER = ["positive", "mixed", "negative", "unclear"]
COLUMNS = ["timestamp", "comment", "rating", "label", "product"]
PRODUCTS = load_products()
# ------------------------------------------------------------------------


def load_feedback():
    if not DATA_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    table = pd.read_csv(DATA_FILE, dtype={"comment": "string", "label": "string", "product": "string"})
    for column in COLUMNS:
        if column not in table:
            table[column] = pd.NA
    table["rating"] = pd.to_numeric(table["rating"], errors="coerce")
    table["product"] = table["product"].fillna("General")
    table["label"] = table["label"].fillna("")
    table["comment"] = table["comment"].fillna("")
    return table[COLUMNS]


def import_demo_feedback(existing, progress_callback=None):
    """Classify bundled demo comments, append only new (comment, rating) pairs."""
    demo = pd.read_csv(DEMO_FILE).fillna({"comment": ""})
    if not {"comment", "rating"}.issubset(demo.columns):
        raise ValueError("Demo CSV must contain comment and rating columns.")

    def signature(comment, rating):
        comment_key = "" if pd.isna(comment) else str(comment).strip().casefold()
        rating_key = None if pd.isna(rating) or str(rating).strip() == "" else int(float(rating))
        return comment_key, rating_key

    seen = {signature(record.comment, record.rating) for record in existing.itertuples(index=False)}
    added = 0
    skipped = 0
    errors = []
    total = len(demo)
    for i, record in enumerate(demo.itertuples(index=False), 1):
        comment = "" if pd.isna(record.comment) else str(record.comment).strip()
        rating = pd.to_numeric(record.rating, errors="coerce")
        if not pd.isna(rating) and (rating != int(rating) or not 1 <= rating <= 5):
            errors.append(f"Row {i}: invalid star rating")
            continue
        if not comment and pd.isna(rating):
            errors.append(f"Row {i}: blank comment and rating")
            continue
        key = signature(comment, rating)
        if key in seen:
            skipped += 1
        else:
            try:
                label, product = label_comment(comment) if comment else ("", "General")
                new_row = pd.DataFrame([{
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "comment": comment,
                    "rating": None if pd.isna(rating) else int(rating),
                    "label": label,
                    "product": product,
                }])
                existing = pd.concat([existing, new_row], ignore_index=True)
                # Save each successful result, even if a later API request fails.
                existing.to_csv(DATA_FILE, index=False)
                seen.add(key)
                added += 1
            except Exception as exc:
                errors.append(f"Row {i}: {exc}")
                # Auth or rate-limit issues may affect all remaining rows; stop early.
                break
        if progress_callback:
            progress_callback(i / total, added, skipped)
    return added, skipped, errors


def get_review_flags(table):
    """Flag mismatched stars/sentiment, mixed comments, and unclear labels."""
    return (
        ((table["label"] == "positive") & (table["rating"] <= LOW_RATING))
        | ((table["label"] == "negative") & (table["rating"] >= HIGH_RATING))
        | table["label"].isin(["mixed", "unclear"])
    ).fillna(False)


def show_metrics(table):
    a, b, c = st.columns(3)
    a.metric("Feedback received", len(table))
    ratings = table["rating"].dropna()
    b.metric("Average star rating", f"{ratings.mean():.1f} / 5" if len(ratings) else "N/A")
    c.metric("Needs manual review", int(get_review_flags(table).sum()))


def show_feedback_table(table):
    """Show complete comment strings; not truncated Plotly hover labels."""
    display = table.sort_values("timestamp", ascending=False, na_position="last")
    st.dataframe(display, hide_index=True, width="stretch", column_config={
        "comment": st.column_config.TextColumn("Full comment", width="large"),
        "rating": st.column_config.NumberColumn("Stars", format="%d"),
    })
    # Expand every flagged comment in full, including lengthy ones.
    for index, record in display.reset_index(drop=True).iterrows():
        label = record["label"] or "No sentiment"
        with st.expander(f"#{index+1} · {record['product']} · {label} · {record['rating'] if pd.notna(record['rating']) else 'No stars'}"):
            st.write(record["comment"] if str(record["comment"]).strip() else "(Rating only; no written comment)")


def show_sentiment_chart(table):
    comments = table.loc[table["comment"].str.strip().ne("") & table["label"].isin(LABEL_ORDER)]
    if comments.empty:
        st.info("No written comments to chart yet.")
        return
    counts = comments.groupby(["product", "label"]).size().reset_index(name="comments")
    fig = px.bar(
        counts, x="product", y="comments", color="label", barmode="stack",
        category_orders={"label": LABEL_ORDER}, color_discrete_map=COLORS,
        labels={"product": "Bakery product", "comments": "Comments", "label": "Sentiment"},
    )
    fig.update_layout(legend_title_text="Sentiment", xaxis_tickangle=-45, margin=dict(b=15))
    st.plotly_chart(fig, width="stretch")


def show_top_rated(table):
    ratings = table.loc[(table["product"] != "General") & table["rating"].notna()]
    ranking = (ratings.groupby("product")["rating"]
               .agg(average="mean", count="size")
               .query("count >= 3")
               .sort_values(["average", "count", "product"], ascending=[False, False, True])
               .head(5).reset_index())
    if ranking.empty:
        st.info("No products have at least 3 star ratings yet.")
        return
    ranking["display"] = ranking["product"] + " (n=" + ranking["count"].astype(str) + ")"
    fig = px.bar(ranking, x="display", y="average", text="average", range_y=[0, 5],
                 labels={"display": "Product (number of ratings)", "average": "Average stars"})
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(xaxis_tickangle=-35, margin=dict(b=15))
    st.plotly_chart(fig, width="stretch")


st.set_page_config(page_title=TITLE, page_icon="🥐", layout="wide")
st.title("🥐 " + TITLE)
st.caption("Tell us about your visit. We use AI to summarize feedback for the bakery owner.")

df = load_feedback()
tab_customer, tab_owner = st.tabs(["Customer: leave feedback", "Owner: analyze feedback"])

with tab_customer:
    st.subheader("How was your bakery visit?")
    with st.form("feedback_form", clear_on_submit=True):
        comment = st.text_area("Your comment (optional)", placeholder="What did you enjoy or what could improve?")
        stars = st.feedback("stars")  # None until chosen; otherwise 0 through 4
        submitted = st.form_submit_button("Send feedback")

    if submitted:
        comment = comment.strip()
        if not comment and stars is None:
            st.error("Please enter a comment, a star rating, or both.")
        else:
            try:
                # Rating-only feedback has no written comment to classify.
                label, product = label_comment(comment) if comment else ("", "General")
                row = pd.DataFrame([{
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "comment": comment, "rating": stars + 1 if stars is not None else None,
                    "label": label, "product": product,
                }])
                updated = pd.concat([df, row], ignore_index=True)
                updated.to_csv(DATA_FILE, index=False)
                df = load_feedback()
                st.success("Thank you! Your feedback was saved.")
                if comment:
                    st.caption(f"AI classification: {label} · Product: {product}")
            except Exception as exc:
                st.error("Feedback was not saved because labeling or storage failed. Please try again.")
                st.caption(f"Error details: {exc}")

with tab_owner:
    with st.expander("Load sample feedback for this assignment", expanded=df.empty):
        st.write("Import 35 fictional bakery reviews from the bundled demo CSV. "
                 "Written comments are classified by Groq, and existing identical reviews are skipped.")
        st.caption("This uses Groq API calls and may take some time. "
                   "The demo reviews are not real customer feedback.")
        if st.button("Import sample reviews", type="secondary"):
            progress = st.progress(0, text="Preparing demo reviews...")
            try:
                added, skipped, errors = import_demo_feedback(
                    df,
                    lambda fraction, a, b: progress.progress(
                        fraction, text=f"Imported {a} · Skipped {b} duplicates"
                    ),
                )
                df = load_feedback()
                if added:
                    st.success(f"Imported {added} new reviews. Skipped {skipped} duplicates.")
                else:
                    st.info(f"No new reviews added. Skipped {skipped} duplicates.")
                if errors:
                    st.warning("Import stopped or skipped some reviews: " + "; ".join(errors[:3]))
            except Exception as exc:
                st.error(f"Import failed: {exc}")
            finally:
                progress.empty()

    if df.empty:
        st.info("No feedback yet. Add your first entry on the Customer tab.")
    else:
        options = ["General Overview"] + PRODUCTS + ["General"]
        selection = st.selectbox("Dashboard view", options, index=0)
        subset = df if selection == "General Overview" else df.loc[df["product"] == selection]
        st.subheader("All feedback" if selection == "General Overview" else f"Product: {selection}")
        show_metrics(subset)

        if selection == "General Overview":
            st.subheader("Customer sentiment by product")
            st.caption("Stacked counts of positive, mixed, negative, and unclear written comments. General is shown separately from named products.")
            show_sentiment_chart(subset)

            st.subheader("Top 5 highest-rated products")
            st.caption("Average star ratings. **Minimum 3 star ratings per product to qualify**; General is excluded. Rating counts appear beside product names.")
            show_top_rated(subset)

            st.subheader("Top 5 most-reviewed products")
            st.caption("Based on written comments mentioning a product; General and rating-only submissions are excluded.")
            product_comments = subset.loc[subset["comment"].str.strip().ne("") & subset["product"].ne("General")]
            most_reviewed = (product_comments.groupby("product").size().sort_values(ascending=False)
                             .head(5).rename("Written comments").reset_index())
            if most_reviewed.empty:
                st.info("No product-specific written comments yet.")
            else:
                st.dataframe(most_reviewed, hide_index=True, width="stretch")
        else:
            st.subheader("Sentiment breakdown")
            comment_counts = subset.loc[subset["comment"].str.strip().ne("") & subset["label"].isin(LABEL_ORDER), "label"].value_counts()
            if comment_counts.empty:
                st.info("No written comments for this product yet.")
            else:
                st.dataframe(pd.DataFrame({"Sentiment": LABEL_ORDER, "Comments": [int(comment_counts.get(x, 0)) for x in LABEL_ORDER]}), hide_index=True, width="stretch")

        st.subheader("Comments needing manual review")
        st.caption("Mixed or unclear sentiment, positive text with 1–2 stars, or negative text with 4–5 stars. Rating-only entries are not automatically flagged.")
        review = subset.loc[get_review_flags(subset)]
        if review.empty:
            st.success("No comments currently need manual review in this view.")
        else:
            show_feedback_table(review)

        with st.expander("All feedback in this view"):
            st.dataframe(subset, hide_index=True, width="stretch")
        st.download_button("Download all feedback (CSV)", df.to_csv(index=False), file_name="feedback.csv", mime="text/csv")
        st.caption("CSV storage is temporary on Streamlit Community Cloud: download feedback to keep it, and do not assume it persists across restarts.")
