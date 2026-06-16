"""
tests/test_topic_modeling.py
=============================
Unit tests for the LDA topic modeling pipeline.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topic_modeling import (
    assign_dominant_topic,
    build_document_term_matrix,
    caterer_topic_breakdown,
    fit_lda,
    get_topic_top_words,
    label_topics,
    preprocess_text,
    topic_sentiment_matrix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    "The food arrived on time and was absolutely delicious, great catering service.",
    "Cold food, late delivery, terrible customer service experience overall.",
    "Excellent value for money, fresh ingredients, will definitely order again.",
    "Driver was rude, wrong order delivered, completely disappointed with service.",
    "Amazing presentation, food was hot and fresh, perfect for our office lunch.",
    "The pasta was overcooked and the salad was wilted when it arrived.",
    "Quick delivery, friendly driver, quality packaging kept everything warm.",
    "Soggy sandwiches, missing items, took 90 minutes for a 30 minute delivery.",
]


@pytest.fixture()
def sample_docs() -> pd.Series:
    return pd.Series(SAMPLE_TEXTS)


@pytest.fixture()
def reviews_df() -> pd.DataFrame:
    return pd.DataFrame({
        "review_id":        range(len(SAMPLE_TEXTS)),
        "caterer_id":       [1, 1, 2, 2, 3, 3, 4, 4],
        "caterer_name":     ["Alpha", "Alpha", "Beta", "Beta",
                             "Gamma", "Gamma", "Delta", "Delta"],
        "cuisine_type":     ["American"] * len(SAMPLE_TEXTS),
        "performance_tier": ["platinum", "platinum", "needs_improvement",
                             "needs_improvement", "gold", "gold",
                             "silver", "silver"],
        "review_date":      pd.to_datetime(["2024-01-15"] * len(SAMPLE_TEXTS)),
        "star_rating":      [5, 1, 5, 1, 5, 2, 4, 1],
        "review_text":      SAMPLE_TEXTS,
        "word_count":       [len(t.split()) for t in SAMPLE_TEXTS],
        "vader_label":      ["positive", "negative", "positive", "negative",
                             "positive", "negative", "positive", "negative"],
        "vader_compound":   [0.85, -0.75, 0.80, -0.82, 0.90, -0.60, 0.70, -0.80],
    })


# ---------------------------------------------------------------------------
# preprocess_text
# ---------------------------------------------------------------------------

class TestPreprocessText:
    def test_returns_string(self):
        assert isinstance(preprocess_text("Hello world food delivery"), str)

    def test_removes_punctuation(self):
        result = preprocess_text("Great food, excellent!")
        assert "," not in result
        assert "!" not in result

    def test_lowercases(self):
        result = preprocess_text("Excellent FOOD Great SERVICE")
        assert result == result.lower()

    def test_stopwords_removed(self):
        result = preprocess_text("the food was very good and the service")
        tokens = result.split()
        # Common stopwords should not appear
        assert "the" not in tokens
        assert "and" not in tokens
        assert "was" not in tokens

    def test_short_tokens_removed(self):
        result = preprocess_text("food is so very good a b")
        tokens = result.split()
        assert all(len(t) > 2 for t in tokens)

    def test_empty_string_returns_empty(self):
        assert preprocess_text("") == ""

    def test_only_stopwords_returns_empty(self):
        result = preprocess_text("the a is")
        assert result.strip() == ""


# ---------------------------------------------------------------------------
# build_document_term_matrix
# ---------------------------------------------------------------------------

class TestBuildDocumentTermMatrix:
    def test_returns_tuple_of_two(self, sample_docs):
        clean = sample_docs.apply(preprocess_text)
        dtm, vectorizer = build_document_term_matrix(clean)
        assert dtm is not None
        assert vectorizer is not None

    def test_dtm_shape_matches_docs(self, sample_docs):
        clean = sample_docs.apply(preprocess_text)
        dtm, _ = build_document_term_matrix(clean)
        assert dtm.shape[0] == len(sample_docs)

    def test_dtm_has_positive_vocabulary(self, sample_docs):
        clean = sample_docs.apply(preprocess_text)
        dtm, vectorizer = build_document_term_matrix(clean)
        assert dtm.shape[1] > 0


# ---------------------------------------------------------------------------
# fit_lda
# ---------------------------------------------------------------------------

class TestFitLda:
    def test_returns_fitted_model(self, sample_docs):
        clean = sample_docs.apply(preprocess_text)
        dtm, _ = build_document_term_matrix(clean)
        model = fit_lda(dtm, n_topics=3)
        assert model is not None

    def test_model_has_correct_n_components(self, sample_docs):
        clean = sample_docs.apply(preprocess_text)
        dtm, _ = build_document_term_matrix(clean)
        model = fit_lda(dtm, n_topics=3)
        assert model.n_components == 3


# ---------------------------------------------------------------------------
# get_topic_top_words
# ---------------------------------------------------------------------------

class TestGetTopicTopWords:
    def test_returns_dataframe(self, sample_docs):
        clean = sample_docs.apply(preprocess_text)
        dtm, vectorizer = build_document_term_matrix(clean)
        model = fit_lda(dtm, n_topics=3)
        df = get_topic_top_words(model, vectorizer, n_words=5)
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_topic(self, sample_docs):
        clean = sample_docs.apply(preprocess_text)
        dtm, vectorizer = build_document_term_matrix(clean)
        model = fit_lda(dtm, n_topics=3)
        df = get_topic_top_words(model, vectorizer, n_words=5)
        assert len(df) == 3

    def test_has_required_columns(self, sample_docs):
        clean = sample_docs.apply(preprocess_text)
        dtm, vectorizer = build_document_term_matrix(clean)
        model = fit_lda(dtm, n_topics=3)
        df = get_topic_top_words(model, vectorizer, n_words=5)
        assert "topic_id" in df.columns
        assert "top_words" in df.columns


# ---------------------------------------------------------------------------
# assign_dominant_topic + label_topics
# ---------------------------------------------------------------------------

class TestAssignDominantTopic:
    def test_adds_topic_columns(self, reviews_df):
        clean = reviews_df["review_text"].apply(preprocess_text)
        dtm, vectorizer = build_document_term_matrix(clean)
        model = fit_lda(dtm, n_topics=3)
        df = assign_dominant_topic(model, dtm, reviews_df.copy())
        assert "dominant_topic" in df.columns
        assert "topic_probability" in df.columns

    def test_topic_probability_between_0_and_1(self, reviews_df):
        clean = reviews_df["review_text"].apply(preprocess_text)
        dtm, vectorizer = build_document_term_matrix(clean)
        model = fit_lda(dtm, n_topics=3)
        df = assign_dominant_topic(model, dtm, reviews_df.copy())
        assert df["topic_probability"].between(0.0, 1.0).all()

    def test_dominant_topic_within_range(self, reviews_df):
        clean = reviews_df["review_text"].apply(preprocess_text)
        dtm, vectorizer = build_document_term_matrix(clean)
        model = fit_lda(dtm, n_topics=3)
        df = assign_dominant_topic(model, dtm, reviews_df.copy())
        assert df["dominant_topic"].between(0, 2).all()

    def test_label_topics_adds_topic_label(self, reviews_df):
        clean = reviews_df["review_text"].apply(preprocess_text)
        dtm, vectorizer = build_document_term_matrix(clean)
        model = fit_lda(dtm, n_topics=5)
        df = assign_dominant_topic(model, dtm, reviews_df.copy())
        df = label_topics(df)
        assert "topic_label" in df.columns
        assert df["topic_label"].notna().all()


# ---------------------------------------------------------------------------
# topic_sentiment_matrix + caterer_topic_breakdown
# ---------------------------------------------------------------------------

class TestAggregations:
    def _build_labeled_df(self, reviews_df):
        clean = reviews_df["review_text"].apply(preprocess_text)
        dtm, vectorizer = build_document_term_matrix(clean)
        model = fit_lda(dtm, n_topics=5)
        df = assign_dominant_topic(model, dtm, reviews_df.copy())
        return label_topics(df)

    def test_topic_sentiment_matrix_is_dataframe(self, reviews_df):
        df = self._build_labeled_df(reviews_df)
        result = topic_sentiment_matrix(df)
        assert isinstance(result, pd.DataFrame)

    def test_topic_sentiment_matrix_sums_to_100_per_row(self, reviews_df):
        df = self._build_labeled_df(reviews_df)
        result = topic_sentiment_matrix(df)
        row_sums = result.sum(axis=1)
        assert (row_sums - 100.0).abs().max() < 1.0

    def test_caterer_topic_breakdown_is_dataframe(self, reviews_df):
        df = self._build_labeled_df(reviews_df)
        result = caterer_topic_breakdown(df)
        assert isinstance(result, pd.DataFrame)
