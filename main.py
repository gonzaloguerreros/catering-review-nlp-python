"""
main.py
=======
End-to-end runner: generate reviews → sentiment → topic modeling → charts
→ Tableau export → executive summary.

Usage:
    python3 main.py
"""

import os, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from data_generator    import generate_reviews
from sentiment_analysis import (
    score_vader, score_textblob, rating_sentiment_agreement,
    caterer_sentiment_summary, monthly_sentiment_trend,
)
from topic_modeling import (
    preprocess_text, build_document_term_matrix, fit_lda,
    get_topic_top_words, assign_dominant_topic, label_topics,
    topic_sentiment_matrix, caterer_topic_breakdown, TOPIC_LABELS,
)
from visualizations import (
    plot_sentiment_by_tier, plot_wordclouds, plot_sentiment_trend,
    plot_topic_heatmap, plot_net_sentiment_ranking, plot_topic_sentiment,
)


def _section(title):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")


def main():

    # -----------------------------------------------------------------------
    # 1. Generate data
    # -----------------------------------------------------------------------
    _section("STEP 1 — Generating Review Dataset")
    os.makedirs("data",    exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("tableau_export", exist_ok=True)

    df = generate_reviews(n_reviews=1200)
    df.to_csv("data/reviews.csv", index=False)
    print(f"  Generated {len(df):,} reviews across {df.caterer_name.nunique()} caterers")
    print(f"  Date range: {df.review_date.min()} → {df.review_date.max()}")
    print(f"  Avg star rating: {df.star_rating.mean():.2f}")

    # -----------------------------------------------------------------------
    # 2. Sentiment scoring
    # -----------------------------------------------------------------------
    _section("STEP 2 — Sentiment Analysis (VADER + TextBlob)")
    df = score_vader(df)
    df = score_textblob(df)
    print("\n  Sentiment distribution:")
    print(df["vader_label"].value_counts(normalize=True).mul(100).round(1).to_string())

    agree = rating_sentiment_agreement(df)
    print(f"\n  Star-rating ↔ VADER agreement rate: {agree.attrs['agreement_rate']*100:.1f}%")
    print(f"  Cramér's V (association strength):   {agree.attrs['cramers_v']}")

    # -----------------------------------------------------------------------
    # 3. Topic modeling
    # -----------------------------------------------------------------------
    _section("STEP 3 — Topic Modeling (LDA, 5 topics)")
    df["clean_text"] = df["review_text"].apply(preprocess_text)
    dtm, vectorizer  = build_document_term_matrix(df["clean_text"])
    lda_model        = fit_lda(dtm, n_topics=5)

    print("\n  Discovered Topics — Top 10 Words Each:")
    topic_words = get_topic_top_words(lda_model, vectorizer, n_words=10)
    for _, row in topic_words.iterrows():
        label = TOPIC_LABELS.get(row["topic_id"], f"Topic {row['topic_id']}")
        print(f"  [{label}]  {row['top_words']}")

    df = assign_dominant_topic(lda_model, dtm, df)
    df = label_topics(df)

    print("\n  Topic distribution:")
    print(df["topic_label"].value_counts(normalize=True).mul(100).round(1).to_string())

    # -----------------------------------------------------------------------
    # 4. Aggregated metrics
    # -----------------------------------------------------------------------
    _section("STEP 4 — Caterer Sentiment Summary")
    summary      = caterer_sentiment_summary(df)
    trend        = monthly_sentiment_trend(df)
    topic_sent   = topic_sentiment_matrix(df)
    caterer_topics = caterer_topic_breakdown(df)

    print("\n  Top 5 caterers by Net Sentiment Score:")
    print(summary[["caterer_name", "total_reviews", "avg_star_rating",
                   "net_sentiment_score"]].head(5).to_string(index=False))

    print("\n  Bottom 3 caterers by Net Sentiment Score:")
    print(summary[["caterer_name", "total_reviews", "avg_star_rating",
                   "net_sentiment_score"]].tail(3).to_string(index=False))

    print("\n  Topic × Sentiment (% within topic):")
    print(topic_sent.to_string())

    # -----------------------------------------------------------------------
    # 5. Charts
    # -----------------------------------------------------------------------
    _section("STEP 5 — Generating Charts")
    plot_sentiment_by_tier(df)
    plot_wordclouds(df)
    plot_sentiment_trend(trend)
    plot_topic_heatmap(caterer_topics)
    plot_net_sentiment_ranking(summary)
    plot_topic_sentiment(topic_sent)

    # -----------------------------------------------------------------------
    # 6. Tableau export — flat CSV optimised for dashboard use
    # -----------------------------------------------------------------------
    _section("STEP 6 — Tableau Export")

    # Review-level flat file (one row per review — use for filters & drill-down)
    tableau_reviews = df[[
        "review_id", "caterer_id", "caterer_name", "cuisine_type",
        "performance_tier", "review_date", "star_rating",
        "vader_compound", "vader_label", "tb_polarity", "tb_subjectivity",
        "topic_label", "topic_probability", "word_count"
    ]].copy()
    tableau_reviews["review_date"] = pd.to_datetime(tableau_reviews["review_date"])
    tableau_reviews.to_csv("tableau_export/reviews_flat.csv", index=False)

    # Caterer summary (one row per caterer — use for scorecard views)
    summary.to_csv("tableau_export/caterer_summary.csv", index=False)

    # Monthly trend (for line charts)
    trend.to_csv("tableau_export/monthly_trend.csv", index=False)

    # Topic × Sentiment matrix
    topic_sent.to_csv("tableau_export/topic_sentiment_matrix.csv")

    print("  Exported 4 files to tableau_export/")
    print("  Connect Tableau to tableau_export/reviews_flat.csv as the primary source")
    print("  Join caterer_summary.csv on caterer_name for scorecard views")

    # -----------------------------------------------------------------------
    # 7. Executive Summary
    # -----------------------------------------------------------------------
    _section("EXECUTIVE SUMMARY — Review Intelligence Report")

    worst = summary.iloc[-1]
    best  = summary.iloc[0]
    # Most common topic in negative reviews
    neg_reviews   = df[df["vader_label"] == "negative"]
    top_complaint_topic = neg_reviews["topic_label"].value_counts().idxmax()
    top_complaint_pct   = neg_reviews["topic_label"].value_counts(normalize=True).iloc[0] * 100

    print(f"""
  Dataset:        {len(df):,} reviews | {df.caterer_name.nunique()} caterers | 24 months
  Avg star rating:{df.star_rating.mean():.2f} / 5.0
  Sentiment split:{(df.vader_label=='positive').mean()*100:.1f}% positive |
                  {(df.vader_label=='neutral').mean()*100:.1f}% neutral |
                  {(df.vader_label=='negative').mean()*100:.1f}% negative

  Top performer:  {best['caterer_name']}
                  NSS {best['net_sentiment_score']:+.1f} | ⭐ {best['avg_star_rating']:.2f} avg

  Needs attention:{worst['caterer_name']}
                  NSS {worst['net_sentiment_score']:+.1f} | ⭐ {worst['avg_star_rating']:.2f} avg

  #1 complaint topic: {top_complaint_topic}
                      ({top_complaint_pct:.1f}% of all negative reviews)

  Recommendations:
  1. Flag {worst['caterer_name']} for account review — NSS below threshold.
  2. Prioritise operational fix for '{top_complaint_topic}' —
     it drives the highest volume of negative reviews platform-wide.
  3. Share positive review language from {best['caterer_name']} as best-practice
     examples with lower-tier caterers during onboarding coaching.
  4. Use Tableau export to build a live supplier health scorecard — refresh
     monthly to track whether interventions improve sentiment over time.
    """)


if __name__ == "__main__":
    main()
