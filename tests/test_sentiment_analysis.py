"""
tests/test_sentiment_analysis.py
=================================
Unit tests for sentiment_analysis and topic_modeling modules.

Run with::

    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentiment_analysis import (
    caterer_sentiment_summary,
    monthly_sentiment_trend,
    score_textblob,
    score_vader,
)
from topic_modeling import build_document_term_matrix, preprocess_text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_reviews() -> pd.DataFrame:
    """Minimal review DataFrame with clear positive and negative examples."""
    return pd.DataFrame({
        "review_id":        [1, 2, 3, 4, 5],
        "caterer_id":       [1, 1, 2, 2, 3],
        "caterer_name":     ["Alpha"] * 2 + ["Beta"] * 2 + ["Gamma"],
        "cuisine_type":     ["American"] * 5,
        "performance_tier": ["platinum"] * 2 + ["silver"] * 2 + ["needs_improvement"],
        "review_date":      pd.to_datetime(["2024-01-15", "2024-02-10",
                                            "2024-01-20", "2024-03-05",
                                            "2024-02-28"]),
        "star_rating":      [5, 4, 3, 2, 1],
        "review_text":      [
            "Absolutely delicious food, arrived on time, wonderful experience!",
            "Great food, very fresh and well packaged. Will order again.",
            "Decent but nothing special. Hit or miss quality.",
            "Terrible — food arrived cold and 45 minutes late. Very disappointed.",
            "Disgusting, overpriced, and customer service was completely unresponsive.",
        ],
        "word_count":       [8, 9, 8, 10, 9],
    })


# ---------------------------------------------------------------------------
# VADER scoring
# ---------------------------------------------------------------------------

class TestScoreVader:
    def test_adds_expected_columns(self, sample_reviews):
        result = score_vader(sample_reviews)
        for col in ("vader_compound", "vader_positive", "vader_negative",
                    "vader_neutral", "vader_label"):
            assert col in result.columns, f"Missing column: {col}"

    def test_compound_in_range(self, sample_reviews):
        result = score_vader(sample_reviews)
        assert result["vader_compound"].between(-1.0, 1.0).all()

    def test_positive_review_has_positive_label(self, sample_reviews):
        result = score_vader(sample_reviews)
        five_star = result[result["star_rating"] == 5].iloc[0]
        assert five_star["vader_compound"] > 0.05

    def test_negative_review_has_negative_compound(self, sample_reviews):
        result = score_vader(sample_reviews)
        one_star = result[result["star_rating"] == 1].iloc[0]
        # Strong negative language should produce a negative compound score
        assert one_star["vader_compound"] < 0.0

    def test_does_not_mutate_input(self, sample_reviews):
        original_cols = set(sample_reviews.columns)
        score_vader(sample_reviews)
        assert set(sample_reviews.columns) == original_cols


# ---------------------------------------------------------------------------
# TextBlob scoring
# ---------------------------------------------------------------------------

class TestScoreTextblob:
    def test_adds_expected_columns(self, sample_reviews):
        result = score_textblob(sample_reviews)
        for col in ("tb_polarity", "tb_subjectivity", "tb_label"):
            assert col in result.columns

    def test_polarity_in_range(self, sample_reviews):
        result = score_textblob(sample_reviews)
        assert result["tb_polarity"].between(-1.0, 1.0).all()

    def test_subjectivity_in_range(self, sample_reviews):
        result = score_textblob(sample_reviews)
        assert result["tb_subjectivity"].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# Caterer sentiment summary
# ---------------------------------------------------------------------------

class TestCatererSentimentSummary:
    def test_one_row_per_caterer(self, sample_reviews):
        df     = score_vader(sample_reviews)
        df     = score_textblob(df)
        result = caterer_sentiment_summary(df)
        assert len(result) == sample_reviews["caterer_name"].nunique()

    def test_net_sentiment_score_bounds(self, sample_reviews):
        df     = score_vader(sample_reviews)
        df     = score_textblob(df)
        result = caterer_sentiment_summary(df)
        assert result["net_sentiment_score"].between(-100, 100).all()

    def test_total_reviews_correct(self, sample_reviews):
        df     = score_vader(sample_reviews)
        df     = score_textblob(df)
        result = caterer_sentiment_summary(df)
        assert result["total_reviews"].sum() == len(sample_reviews)


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessText:
    def test_returns_string(self):
        assert isinstance(preprocess_text("Hello world!"), str)

    def test_removes_punctuation(self):
        result = preprocess_text("Hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_lowercases_output(self):
        result = preprocess_text("DELICIOUS Food")
        assert result == result.lower()

    def test_empty_string_returns_empty(self):
        assert preprocess_text("") == ""


# ---------------------------------------------------------------------------
# Document-term matrix
# ---------------------------------------------------------------------------

class TestBuildDTM:
    def test_matrix_shape_matches_documents(self, sample_reviews):
        texts = sample_reviews["review_text"].apply(preprocess_text)
        dtm, vectorizer = build_document_term_matrix(texts, min_df=1)
        assert dtm.shape[0] == len(sample_reviews)

    def test_vocabulary_not_empty(self, sample_reviews):
        texts = sample_reviews["review_text"].apply(preprocess_text)
        _dtm, vectorizer = build_document_term_matrix(texts, min_df=1)
        assert len(vectorizer.get_feature_names_out()) > 0
