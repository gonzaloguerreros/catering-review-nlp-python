"""
visualizations.py
=================
Charts and word clouds for the NLP review analysis.
All figures saved to outputs/ as PNGs.
"""


import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colour palette
POS_COLOR  = "#2ECC71"
NEG_COLOR  = "#E74C3C"
NEU_COLOR  = "#95A5A6"
BG_COLOR   = "#F8F9FA"

sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})

def _save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → Saved: {path}")


# ---------------------------------------------------------------------------
# 1. Sentiment Distribution by Performance Tier
# ---------------------------------------------------------------------------

def plot_sentiment_by_tier(df: pd.DataFrame) -> plt.Figure:
    """
    Stacked bar chart showing the proportion of positive / neutral / negative
    reviews for each performance tier.  Validates that the tier labels
    (Platinum/Gold/Silver/Needs Improvement) correlate with actual customer
    sentiment — a sanity check and a story for stakeholders.
    """
    order = ["platinum", "gold", "silver", "needs_improvement"]
    tier_sent = (
        pd.crosstab(df["performance_tier"], df["vader_label"], normalize="index") * 100
    ).reindex(order)

    fig, ax = plt.subplots(figsize=(9, 5))
    # Ensure all three columns exist even if a sentiment class has zero reviews
    for col in ["positive", "neutral", "negative"]:
        if col not in tier_sent.columns:
            tier_sent[col] = 0.0
    tier_sent[["positive", "neutral", "negative"]].plot(
        kind="bar", stacked=True, ax=ax,
        color=[POS_COLOR, NEU_COLOR, NEG_COLOR], width=0.55
    )
    ax.set_xlabel("Performance Tier")
    ax.set_ylabel("Share of Reviews (%)")
    ax.set_title("Review Sentiment by Caterer Performance Tier")
    ax.set_xticklabels(
        ["Platinum", "Gold", "Silver", "Needs Improvement"], rotation=0
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(["Positive", "Neutral", "Negative"], loc="upper right")
    fig.tight_layout()
    _save(fig, "01_sentiment_by_tier.png")
    return fig


# ---------------------------------------------------------------------------
# 2. Word Cloud — Positive vs. Negative Reviews
# ---------------------------------------------------------------------------

def plot_wordclouds(df: pd.DataFrame) -> plt.Figure:
    """
    Side-by-side word clouds for positive and negative review corpora.
    Word size is proportional to TF (term frequency) within the sentiment class.
    Provides a quick visual vocabulary comparison — instantly shows what
    positive reviewers talk about vs. what complainers focus on.
    """
    pos_text = " ".join(df.loc[df["vader_label"] == "positive", "review_text"])
    neg_text = " ".join(df.loc[df["vader_label"] == "negative", "review_text"])

    wc_pos = WordCloud(
        width=700, height=350, background_color="white",
        colormap="Greens", max_words=80, collocations=False
    ).generate(pos_text)

    wc_neg = WordCloud(
        width=700, height=350, background_color="white",
        colormap="Reds", max_words=80, collocations=False
    ).generate(neg_text)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(wc_pos, interpolation="bilinear")
    axes[0].axis("off")
    axes[0].set_title("Positive Reviews — Top Words", fontsize=13, fontweight="bold")

    axes[1].imshow(wc_neg, interpolation="bilinear")
    axes[1].axis("off")
    axes[1].set_title("Negative Reviews — Top Words", fontsize=13, fontweight="bold")

    fig.suptitle("Word Clouds: Positive vs. Negative Review Language",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "02_wordclouds.png")
    return fig


# ---------------------------------------------------------------------------
# 3. Monthly Sentiment Trend for Top 5 Caterers
# ---------------------------------------------------------------------------

def plot_sentiment_trend(trend_df: pd.DataFrame) -> plt.Figure:
    """
    Line chart of 3-month rolling VADER score for the top 5 caterers by volume.
    Shows whether caterer sentiment is improving or declining — the primary
    signal for proactive account management.
    """
    top5 = (
        trend_df.groupby("caterer_name")["review_count"]
                .sum()
                .nlargest(5)
                .index
    )
    df_top = trend_df[trend_df["caterer_name"].isin(top5)].copy()
    df_top["review_month"] = pd.to_datetime(df_top["review_month"])

    fig, ax = plt.subplots(figsize=(12, 5))
    palette = sns.color_palette("tab10", len(top5))

    for (name, grp), color in zip(df_top.groupby("caterer_name"), palette):
        grp = grp.sort_values("review_month")
        ax.plot(grp["review_month"], grp["vader_3mo_avg"],
                label=name, color=color, linewidth=2)
        ax.fill_between(grp["review_month"], grp["vader_3mo_avg"],
                        alpha=0.07, color=color)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_xlabel("Month")
    ax.set_ylabel("VADER Compound Score (3-mo rolling avg)")
    ax.set_title("Monthly Sentiment Trend — Top 5 Caterers by Review Volume")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    _save(fig, "03_sentiment_trend.png")
    return fig


# ---------------------------------------------------------------------------
# 4. Topic Distribution — Heatmap
# ---------------------------------------------------------------------------

def plot_topic_heatmap(caterer_topics: pd.DataFrame) -> plt.Figure:
    """
    Heatmap of topic share (%) per caterer.
    Allows at-a-glance comparison: which caterers have the most delivery
    complaints, which have the most food quality praise.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(
        caterer_topics, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, cbar_kws={"label": "% of Reviews"}
    )
    ax.set_title("Review Topic Distribution by Caterer (%)", pad=15)
    ax.set_xlabel("Topic")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, "04_topic_heatmap.png")
    return fig


# ---------------------------------------------------------------------------
# 5. Net Sentiment Score — Caterer Ranking
# ---------------------------------------------------------------------------

def plot_net_sentiment_ranking(summary_df: pd.DataFrame) -> plt.Figure:
    """
    Horizontal bar chart ranking caterers by Net Sentiment Score (NSS).
    NSS = % positive reviews − % negative reviews.
    Colour-coded by performance tier.
    """
    df = summary_df.sort_values("net_sentiment_score")

    tier_colors = {
        "platinum":          "#2ECC71",
        "gold":              "#F1C40F",
        "silver":            "#95A5A6",
        "needs_improvement": "#E74C3C",
    }
    colors = [tier_colors.get(t, "#333") for t in df["performance_tier"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(df["caterer_name"], df["net_sentiment_score"],
                   color=colors, height=0.6, zorder=3)

    # Annotate with NSS value
    for bar, val in zip(bars, df["net_sentiment_score"]):
        x = bar.get_width() + 0.5 if val >= 0 else bar.get_width() - 0.5
        ax.text(x, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}", va="center", fontsize=9)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Net Sentiment Score (%positive − %negative)")
    ax.set_title("Caterer Net Sentiment Score Ranking\n(colour = performance tier)")

    # Legend for tier colours
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t.replace("_", " ").title())
                       for t, c in tier_colors.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, "05_net_sentiment_ranking.png")
    return fig


# ---------------------------------------------------------------------------
# 6. Topic × Sentiment Breakdown
# ---------------------------------------------------------------------------

def plot_topic_sentiment(topic_sent_matrix: pd.DataFrame) -> plt.Figure:
    """
    Grouped bar chart showing sentiment breakdown within each topic.
    Directly answers: 'Which topics drive the most negative reviews?'
    """
    # Extract positive/neutral/negative columns
    plot_df = topic_sent_matrix.copy()
    for col in ["positive", "neutral", "negative"]:
        if col not in plot_df.columns:
            plot_df[col] = 0.0
    plot_df = plot_df[["positive", "neutral", "negative"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", ax=ax,
                 color=[POS_COLOR, NEU_COLOR, NEG_COLOR], width=0.65)
    ax.set_xlabel("")
    ax.set_ylabel("Share of Topic Reviews (%)")
    ax.set_title("Sentiment Distribution Within Each Review Topic")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(["Positive", "Neutral", "Negative"])
    fig.tight_layout()
    _save(fig, "06_topic_sentiment.png")
    return fig
