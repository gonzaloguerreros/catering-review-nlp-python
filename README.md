# Catering Marketplace Review Intelligence — NLP Analysis (Python)

**Role context:** Product Analyst portfolio project demonstrating NLP techniques applied to supplier quality monitoring on a B2B catering marketplace. Combines sentiment analysis, topic modeling, and actionable business recommendations.

---

## Business Context

A B2B catering marketplace (modelled after ezCater) collects post-delivery reviews from corporate accounts. With 15+ active caterers and thousands of reviews per year, manually reading reviews to identify quality issues is not scalable. This project builds an automated review intelligence pipeline that:

1. **Scores** every review for sentiment
2. **Discovers** what customers are actually talking about (topics)
3. **Ranks** caterers by a Net Sentiment Score (NSS)
4. **Exports** Tableau-ready files for a live supplier health dashboard

---

## Analytical Pipeline

```
Raw Reviews (1,200)
     │
     ▼
VADER Sentiment Scoring        ← rule-based, fast, no training needed
TextBlob Polarity/Subjectivity ← adds objectivity signal
     │
     ▼
Text Preprocessing             ← lowercase, de-punctuate, stop words, lemmatise
     │
     ▼
LDA Topic Modeling (5 topics)  ← unsupervised discovery of review themes
     │
     ▼
Caterer Scorecard              ← NSS, avg VADER score, topic breakdown per caterer
     │
     ▼
Tableau Export + Charts        ← 6 visualisations + 4 CSV files for dashboard
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Reviews analysed | 1,200 |
| Caterers covered | 15 |
| Date range | Jan 2023 – Dec 2024 |
| Positive sentiment | 95.2% |
| Star–sentiment agreement (Cramér's V) | 0.62 (strong) |

**Top performer:** Bella Cucina Events — NSS +97.9, ⭐ 4.93 avg  
**Needs attention:** Federal Street Fare — NSS +79.7, ⭐ 3.93 avg  
**#1 complaint driver:** Packaging & Presentation (33% of negative reviews)

---

## NLP Methods

| Technique | Tool | Why Used |
|-----------|------|----------|
| VADER sentiment scoring | `nltk.sentiment.vader` | Purpose-built for short reviews; outputs compound score in [-1,+1]; no training data needed |
| TextBlob polarity + subjectivity | `textblob` | Adds subjectivity dimension — high-subjectivity reviews are emotionally driven, which signals urgency |
| Text preprocessing | `nltk`, regex | Lowercase, remove punctuation, stop word removal, lemmatisation — required to reduce vocabulary noise before LDA |
| LDA topic modeling | `sklearn.decomposition` | Unsupervised; each review is a probability mixture of topics; discovers themes without pre-labelling |
| CountVectorizer (DTM) | `sklearn` | Converts text to bag-of-words with bigrams; `min_df`/`max_df` filter rare/universal words |
| Cramér's V | `scipy.stats` | Measures association between star rating (ordinal) and sentiment label (categorical) — better than Pearson for non-continuous data |

---

## Project Structure

```
catering-review-nlp-python/
├── src/
│   ├── data_generator.py      # 1,200 synthetic reviews with tier-based quality variation
│   ├── sentiment_analysis.py  # VADER + TextBlob scoring, agreement check, NSS
│   ├── topic_modeling.py      # Text preprocessing, DTM, LDA fit, topic assignment
│   └── visualizations.py     # 6 charts (tier sentiment, word clouds, trend, heatmap, ranking, topic)
├── data/
│   └── reviews.csv
├── outputs/
│   ├── 01_sentiment_by_tier.png
│   ├── 02_wordclouds.png
│   ├── 03_sentiment_trend.png
│   ├── 04_topic_heatmap.png
│   ├── 05_net_sentiment_ranking.png
│   └── 06_topic_sentiment.png
├── tableau_export/
│   ├── reviews_flat.csv           # One row per review — primary Tableau source
│   ├── caterer_summary.csv        # Caterer scorecard
│   ├── monthly_trend.csv          # For line charts
│   └── topic_sentiment_matrix.csv
├── main.py
└── requirements.txt
```

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

## Tableau Dashboard Setup

1. Open Tableau Desktop
2. Connect to `tableau_export/reviews_flat.csv`
3. Add relationship to `caterer_summary.csv` on `caterer_name`
4. Build views:
   - **Scorecard:** NSS + avg star rating per caterer (use caterer_summary)
   - **Trend line:** VADER 3-month rolling average over time (use monthly_trend.csv)
   - **Topic heatmap:** % reviews per topic per caterer (pivot reviews_flat)
   - **Drill-down filter:** Filter by tier, industry, date range on reviews_flat

---

## Design Decisions

- **Why VADER over a transformer model (BERT)?** VADER is fully interpretable, runs on a laptop without a GPU, and was specifically designed for short consumer reviews. For a portfolio project — and for most product analytics work — VADER's accuracy is sufficient and its speed and interpretability are advantages.
- **Why LDA over BERTopic?** Same rationale — LDA is the industry standard, widely understood, and produces clean word distributions. BERTopic would require a GPU and additional dependencies.
- **Why lemmatisation over stemming?** LDA's output is shown as top words to stakeholders. Stemmed words ('delici', 'arri') are unreadable; lemmatised words ('delicious', 'arrive') are not.
- **Why Net Sentiment Score?** NSS mirrors the NPS (Net Promoter Score) framework that product and customer success teams already use. It reduces a complex sentiment distribution to a single trackable number.

---

*Dataset is fully synthetic. All reviews and figures are generated programmatically for portfolio demonstration purposes.*
