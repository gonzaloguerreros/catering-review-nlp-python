"""
topic_modeling.py
=================
Unsupervised topic discovery using Latent Dirichlet Allocation (LDA).

What is LDA?
------------
LDA treats each document (review) as a mixture of latent topics, and each
topic as a probability distribution over words.  The model is trained
unsupervised — it discovers the topics from word co-occurrence patterns
without being told what the topics are.

For a catering marketplace, we expect LDA to surface topics like:
  - Food quality (fresh, flavor, delicious, taste)
  - Delivery & logistics (arrived, late, time, driver)
  - Packaging & presentation (container, label, presentation)
  - Value & pricing (price, worth, expensive, affordable)
  - Customer service (responsive, communication, helpful)

Why LDA over simpler methods?
------------------------------
- Keyword search: requires knowing the topics in advance (no discovery)
- Clustering (k-means on TF-IDF): hard document assignments miss mixed topics
- LDA: probabilistic, handles reviews that are about multiple topics,
  and produces interpretable word distributions per topic

Implementation uses scikit-learn's LatentDirichletAllocation, which is
well-documented, production-stable, and does not require a GPU.
"""



import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Download NLTK resources required for preprocessing
nltk.download("stopwords",    quiet=True)
nltk.download("wordnet",      quiet=True)
nltk.download("omw-1.4",      quiet=True)
nltk.download("punkt",        quiet=True)
nltk.download("punkt_tab",    quiet=True)

# ---------------------------------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------------------------------

# Domain-specific stop words to add on top of NLTK's English list
CATERING_STOPWORDS = {
    "food", "order", "catering", "caterer", "delivery", "ordered",
    "team", "office", "lunch", "meal", "company", "get", "got",
    "also", "even", "really", "would", "could", "us", "our",
}

def preprocess_text(text: str, lemmatizer: WordNetLemmatizer | None = None) -> str:
    """
    Clean and normalise a single review string.

    Pipeline:
      1. Lowercase
      2. Remove punctuation and digits (not semantically useful for topics)
      3. Tokenise
      4. Remove stop words (NLTK English + catering-domain extras)
      5. Lemmatise (reduce words to root form: 'arriving' → 'arrive')

    Lemmatisation is preferred over stemming here because:
    - Stemming is aggressive and creates non-words ('delicious' → 'delici')
    - LDA topic labels are shown as top words — they need to be readable
    """
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()

    stop_words = set(stopwords.words("english")) | CATERING_STOPWORDS

    # Step 1-2: lowercase and strip punctuation/digits
    text = text.lower()
    text = re.sub(r"[%s\d]" % re.escape(string.punctuation), " ", text)

    # Step 3-4: tokenise and remove stop words
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 2]

    # Step 5: lemmatise each token
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def build_document_term_matrix(texts: pd.Series,
                                max_features: int = 500,
                                min_df: int = 3,
                                max_df: float = 0.90) -> tuple:
    """
    Convert preprocessed text into a document-term matrix (DTM).

    Parameters
    ----------
    texts        : Series of preprocessed review strings
    max_features : vocabulary size cap (top N words by frequency)
    min_df       : ignore words appearing in fewer than min_df documents
                   (removes typos and one-off phrases)
    max_df       : ignore words in more than max_df fraction of documents
                   (removes words so common they carry no topic signal)

    Returns
    -------
    (dtm, vectorizer) — the sparse matrix and the fitted CountVectorizer
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),   # include bigrams (e.g., "on time", "fresh food")
    )
    dtm = vectorizer.fit_transform(texts)
    return dtm, vectorizer


# ---------------------------------------------------------------------------
# LDA Model
# ---------------------------------------------------------------------------

def fit_lda(dtm, n_topics: int = 5, random_state: int = 42) -> LatentDirichletAllocation:
    """
    Fit a Latent Dirichlet Allocation model.

    Parameters
    ----------
    dtm        : document-term matrix from build_document_term_matrix
    n_topics   : number of latent topics to discover.
                 5 is a reasonable starting point for a catering review corpus
                 — we expect food quality, delivery, packaging, value, service.
                 In production, n_topics is tuned using perplexity or coherence score.
    random_state : for reproducibility

    Returns
    -------
    Fitted LDA model
    """
    lda = LatentDirichletAllocation(
        n_components  = n_topics,
        max_iter      = 20,          # sufficient for a small corpus
        learning_method = "batch",   # batch is more stable than online for <10K docs
        random_state  = random_state,
        n_jobs        = -1,          # use all CPU cores
    )
    lda.fit(dtm)
    return lda


def get_topic_top_words(lda: LatentDirichletAllocation,
                         vectorizer: CountVectorizer,
                         n_words: int = 10) -> pd.DataFrame:
    """
    Extract the top N words for each discovered topic.

    These word lists are what you present to stakeholders to name each topic.
    For example, if topic 2's top words are ['late', 'arrived', 'delayed',
    'time', 'driver'], you would label it 'Delivery & Logistics'.
    """
    feature_names = vectorizer.get_feature_names_out()
    rows = []
    for topic_idx, topic_weights in enumerate(lda.components_):
        # Sort words by weight (highest = most representative of the topic)
        top_indices = topic_weights.argsort()[::-1][:n_words]
        top_words   = [feature_names[i] for i in top_indices]
        rows.append({
            "topic_id":   topic_idx,
            "top_words":  ", ".join(top_words),
        })
    return pd.DataFrame(rows)


def assign_dominant_topic(lda: LatentDirichletAllocation,
                           dtm,
                           df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each review its dominant topic and topic probability.

    The dominant topic is the one with the highest probability in the
    document's topic mixture.  Also records the probability — low-confidence
    assignments (e.g., dominant topic at 25%) signal genuinely mixed reviews.
    """
    # Transform: get topic probability distribution for every document
    doc_topic_matrix = lda.transform(dtm)   # shape: (n_reviews, n_topics)

    df = df.copy()
    df["dominant_topic"]   = doc_topic_matrix.argmax(axis=1)
    df["topic_probability"] = doc_topic_matrix.max(axis=1).round(4)

    # Store the full distribution as a string (useful for debugging)
    df["topic_distribution"] = [
        str([round(p, 3) for p in row]) for row in doc_topic_matrix
    ]
    return df


# Human-readable topic labels (assigned after inspecting top words)
TOPIC_LABELS = {
    0: "Food Quality & Taste",
    1: "Delivery & Timeliness",
    2: "Packaging & Presentation",
    3: "Value & Pricing",
    4: "Customer Service",
}


def label_topics(df: pd.DataFrame,
                  topic_labels: dict = TOPIC_LABELS) -> pd.DataFrame:
    """Map numeric topic IDs to human-readable labels."""
    df = df.copy()
    df["topic_label"] = df["dominant_topic"].map(topic_labels).fillna("Other")
    return df


# ---------------------------------------------------------------------------
# Topic × Sentiment Cross-Analysis
# ---------------------------------------------------------------------------

def topic_sentiment_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-tabulate topic distribution against sentiment labels.

    This answers: 'Which topics are most associated with negative reviews?'
    For example, if 70% of Delivery-topic reviews are negative, that is a
    clear operational signal to route to the caterer's account manager.
    """
    matrix = pd.crosstab(
        df["topic_label"],
        df["vader_label"],
        values=df["vader_compound"],
        aggfunc="count",
        normalize="index"     # row-normalise → proportions within each topic
    ).round(3) * 100

    # Add avg VADER score per topic
    avg_score = df.groupby("topic_label")["vader_compound"].mean().round(3)
    matrix["avg_vader_score"] = avg_score

    return matrix.sort_values("avg_vader_score")


def caterer_topic_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each caterer, show what percentage of their reviews fall into each topic.

    Useful for targeted supplier coaching: if a caterer has 40% of reviews
    in the Delivery topic but a low sentiment score on those reviews, the
    intervention is logistics — not food quality.
    """
    breakdown = pd.crosstab(
        df["caterer_name"],
        df["topic_label"],
        normalize="index"
    ).round(3) * 100

    return breakdown
