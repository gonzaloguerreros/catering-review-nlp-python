"""
config.py
=========
Single source of truth for all dataset parameters, NLP hyperparameters,
and file paths used across the catering review intelligence package.

Centralising configuration follows the Twelve-Factor App methodology
(factor III: config) and Google Python Style Guide §2.7.  Hyperparameters
for machine learning models belong here so they are easy to tune without
hunting through source files.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root and derived paths (pathlib — PEP 428)
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path         = ROOT_DIR / "data"
OUTPUTS_DIR: Path      = ROOT_DIR / "outputs"
TABLEAU_DIR: Path      = ROOT_DIR / "tableau_export"

for _dir in (DATA_DIR, OUTPUTS_DIR, TABLEAU_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset generation parameters
# ---------------------------------------------------------------------------

#: Total number of synthetic reviews to generate.
N_REVIEWS: int = 1_200

#: Random seed for reproducible dataset generation.
RANDOM_SEED: int = 7

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

#: Domain-specific stop words appended to NLTK's English list.
#: These words are so common in catering reviews that they carry no
#: discriminative signal for topic modeling.
CATERING_STOPWORDS: frozenset[str] = frozenset({
    "food", "order", "catering", "caterer", "delivery", "ordered",
    "team", "office", "lunch", "meal", "company", "get", "got",
    "also", "even", "really", "would", "could", "us", "our",
})

# ---------------------------------------------------------------------------
# Document-term matrix (CountVectorizer) hyperparameters
# ---------------------------------------------------------------------------

#: Maximum vocabulary size — top N words by corpus frequency.
DTM_MAX_FEATURES: int = 500

#: Minimum document frequency — ignore words in fewer than this many reviews.
#: Removes typos and one-off phrases that won't generalise to any topic.
DTM_MIN_DF: int = 3

#: Maximum document frequency — ignore words in more than this fraction of
#: reviews.  Words that appear everywhere carry no discriminative power.
DTM_MAX_DF: float = 0.90

#: Include unigrams and bigrams (e.g., "on time", "fresh food").
DTM_NGRAM_RANGE: tuple[int, int] = (1, 2)

# ---------------------------------------------------------------------------
# LDA topic model hyperparameters
# ---------------------------------------------------------------------------

#: Number of latent topics.  Chosen based on domain knowledge (food quality,
#: delivery, packaging, value, customer service).  In production, tune using
#: coherence score over a range of n_components.
LDA_N_TOPICS: int = 5

#: Maximum EM iterations.  20 is sufficient for a corpus of < 10 K documents
#: with a stable vocabulary.
LDA_MAX_ITER: int = 20

#: Learning method.  'batch' (full-dataset EM) is more stable than 'online'
#: for small corpora; 'online' is preferred for streaming / large corpora.
LDA_LEARNING_METHOD: str = "batch"

#: Random seed for LDA initialisation.
LDA_RANDOM_STATE: int = 42

#: Number of top words to extract per topic for display.
LDA_TOP_N_WORDS: int = 10

#: Human-assigned topic labels derived from inspecting the top-word lists.
#: Update these after each model run if the top words shift.
TOPIC_LABELS: dict[int, str] = {
    0: "Food Quality & Taste",
    1: "Delivery & Timeliness",
    2: "Packaging & Presentation",
    3: "Value & Pricing",
    4: "Customer Service",
}

# ---------------------------------------------------------------------------
# Sentiment analysis thresholds (VADER)
# ---------------------------------------------------------------------------

#: VADER compound score boundaries (Hutto & Gilbert, 2014).
#: compound >= VADER_POS_THRESHOLD  → positive
#: compound <= VADER_NEG_THRESHOLD  → negative
#: otherwise                        → neutral
VADER_POS_THRESHOLD: float =  0.05
VADER_NEG_THRESHOLD: float = -0.05

# ---------------------------------------------------------------------------
# Statistical thresholds
# ---------------------------------------------------------------------------
ALPHA: float = 0.05
