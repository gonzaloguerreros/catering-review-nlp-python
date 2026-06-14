"""
sentiment_analysis.py
=====================
Sentiment scoring and trend analysis for catering marketplace reviews.

Two complementary approaches are used:

1. VADER (Valence Aware Dictionary and sEntiment Reasoner)
   - Rule-based lexicon designed specifically for short, informal text
     (user reviews, social media, ratings)
   - Outputs a compound score in [-1, +1]: negative < -0.05, positive > 0.05
   - Fast, interpretable, requires no training data
   - Best for: classifying each review as positive/neutral/negative

2. TextBlob
   - Pattern-based NLP library with polarity [-1,+1] and subjectivity [0,1]
   - Subjectivity score adds an extra signal: high-subjectivity reviews tend
     to be emotionally driven, low-subjectivity reviews tend to be factual
   - Best for: understanding how emotional vs. objective the language is

Both scores are computed so we can cross-validate and use each where it
is strongest — VADER for classification, TextBlob for subjectivity.
"""



import nltk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download VADER lexicon on first run (small file, ~1MB)
nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt",         quiet=True)
nltk.download("stopwords",     quiet=True)


# ---------------------------------------------------------------------------
# VADER Sentiment Scoring
# ---------------------------------------------------------------------------

def score_vader(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """
    Apply VADER sentiment scoring to every review.

    Adds columns:
        vader_compound   : overall sentiment score [-1, +1]
        vader_positive   : proportion of text that is positive (0-1)
        vader_negative   : proportion of text that is negative (0-1)
        vader_neutral    : proportion of text that is neutral (0-1)
        vader_label      : 'positive' | 'neutral' | 'negative'

    The compound score thresholds (±0.05) come from the original VADER paper
    (Hutto & Gilbert, 2014) and are the standard for binary/ternary classification.
    """
    sia = SentimentIntensityAnalyzer()
    scores = df[text_col].apply(lambda text: sia.polarity_scores(text))

    df = df.copy()
    df["vader_compound"]  = scores.apply(lambda s: s["compound"])
    df["vader_positive"]  = scores.apply(lambda s: s["pos"])
    df["vader_negative"]  = scores.apply(lambda s: s["neg"])
    df["vader_neutral"]   = scores.apply(lambda s: s["neu"])

    # Classify using VADER's recommended thresholds
    df["vader_label"] = pd.cut(
        df["vader_compound"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negative", "neutral", "positive"]
    )
    return df


# ---------------------------------------------------------------------------
# TextBlob Sentiment + Subjectivity
# ---------------------------------------------------------------------------

def score_textblob(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """
    Apply TextBlob polarity and subjectivity scoring.

    Adds columns:
        tb_polarity      : sentiment polarity [-1, +1]
        tb_subjectivity  : how subjective/emotional the text is [0, 1]
                           0 = purely factual, 1 = purely opinionated
        tb_label         : 'positive' | 'neutral' | 'negative'
    """
    df = df.copy()
    blobs = df[text_col].apply(lambda t: TextBlob(t))

    df["tb_polarity"]     = blobs.apply(lambda b: round(b.sentiment.polarity,     4))
    df["tb_subjectivity"] = blobs.apply(lambda b: round(b.sentiment.subjectivity, 4))

    # Use same ±0.05 threshold for consistency with VADER
    df["tb_label"] = pd.cut(
        df["tb_polarity"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negative", "neutral", "positive"]
    )
    return df


# ---------------------------------------------------------------------------
# Sentiment–Star Rating Agreement Check
# ---------------------------------------------------------------------------

def rating_sentiment_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Measure how well the VADER label agrees with the star rating.

    A review rated 1-2 stars should be negative; 4-5 stars should be positive.
    Disagreements (e.g., 5-star review with negative VADER compound) are
    valuable quality signals — they may indicate sarcasm, short reviews, or
    data quality issues.

    Returns a crosstab of star_rating × vader_label with a Cramér's V
    correlation coefficient measuring overall agreement strength.
    """
    # Map star ratings to expected sentiment
    df = df.copy()
    df["expected_sentiment"] = pd.cut(
        df["star_rating"],
        bins=[0, 2, 3, 5],
        labels=["negative", "neutral", "positive"]
    )

    # Agreement flag
    df["sentiment_agrees"] = df["vader_label"] == df["expected_sentiment"]

    # Cramér's V — measures association between two categorical variables
    # Range: 0 (no association) to 1 (perfect association)
    contingency = pd.crosstab(df["star_rating"], df["vader_label"])
    chi2 = _cramers_v_chi2(contingency.values)
    n    = contingency.values.sum()
    k    = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 else 0.0

    result = pd.crosstab(
        df["star_rating"], df["vader_label"],
        margins=True, margins_name="Total"
    )
    result.attrs["cramers_v"]  = round(cramers_v, 4)
    result.attrs["agreement_rate"] = round(df["sentiment_agrees"].mean(), 4)
    return result


def _cramers_v_chi2(contingency_values: np.ndarray) -> float:
    """Compute chi-squared statistic for a contingency table."""
    from scipy.stats import chi2_contingency
    chi2, _, _, _ = chi2_contingency(contingency_values)
    return chi2


# ---------------------------------------------------------------------------
# Caterer-Level Sentiment Summary
# ---------------------------------------------------------------------------

def caterer_sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment metrics at the caterer level.

    Outputs one row per caterer with:
    - Review volume and star rating distribution
    - Mean VADER compound score
    - Net Sentiment Score (NSS): % positive - % negative reviews
      (analogous to NPS — a single number executives can track)
    - Mean subjectivity score (flags emotionally charged feedback)
    - Month-over-month sentiment trend (is the caterer improving?)
    """
    summary = df.groupby(["caterer_id", "caterer_name", "performance_tier"]).agg(
        total_reviews    = ("review_id",       "count"),
        avg_star_rating  = ("star_rating",     "mean"),
        avg_vader_score  = ("vader_compound",  "mean"),
        pct_positive     = ("vader_label",     lambda x: (x == "positive").mean() * 100),
        pct_neutral      = ("vader_label",     lambda x: (x == "neutral").mean() * 100),
        pct_negative     = ("vader_label",     lambda x: (x == "negative").mean() * 100),
        avg_subjectivity = ("tb_subjectivity", "mean"),
    ).reset_index()

    # Net Sentiment Score: single headline metric (mirrors NPS logic)
    summary["net_sentiment_score"] = (
        summary["pct_positive"] - summary["pct_negative"]
    ).round(1)

    # Round floats for readability
    float_cols = ["avg_star_rating", "avg_vader_score", "avg_subjectivity",
                  "pct_positive", "pct_neutral", "pct_negative"]
    summary[float_cols] = summary[float_cols].round(2)

    return summary.sort_values("net_sentiment_score", ascending=False)


# ---------------------------------------------------------------------------
# Monthly Sentiment Trend
# ---------------------------------------------------------------------------

def monthly_sentiment_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly average sentiment per caterer — enables trend detection.

    Outputs are suitable for a Tableau line chart showing whether a caterer's
    sentiment is improving or deteriorating over time.
    """
    df = df.copy()
    df["review_month"] = pd.to_datetime(df["review_date"]).dt.to_period("M")

    trend = df.groupby(["caterer_name", "review_month"]).agg(
        review_count    = ("review_id",      "count"),
        avg_vader       = ("vader_compound", "mean"),
        avg_star        = ("star_rating",    "mean"),
        pct_negative    = ("vader_label",    lambda x: (x == "negative").mean() * 100),
    ).reset_index()

    trend["review_month"] = trend["review_month"].astype(str)

    # 3-month rolling average per caterer to smooth noise
    trend = trend.sort_values(["caterer_name", "review_month"])
    trend["vader_3mo_avg"] = (
        trend.groupby("caterer_name")["avg_vader"]
             .transform(lambda x: x.rolling(3, min_periods=1).mean())
             .round(4)
    )

    return trend
