# Changelog

All notable changes to this project are documented here.

## [1.2.0] - 2026-06-15
### Changed
- Upgraded to Python 3.11+ (dropped EOL Python 3.8)
- Updated all dependencies: pandas 3.0.3, numpy 2.4.6, scipy 1.17.1,
  scikit-learn 1.9.0, nltk 3.9.4, textblob 0.20.0, wordcloud 1.9.6
- CI matrix updated to Python 3.11 and 3.12

### Fixed
- Removed unused `numpy` imports from `topic_modeling.py` and `visualizations.py`
- Fixed bare f-strings without placeholders in `main.py`
- Removed assigned-but-unused `top_neg_topic` variable

## [1.1.0] - 2026-06-07
### Fixed
- VADER scoring 0% negative reviews: rewrote negative seed sentences with
  explicit strong-sentiment language (VADER requires clear valence markers)
- Missing `negative` column in crosstab: added zero-fill for sentiment labels
  not present in the data to prevent KeyError on all-positive corpora
- GitHub Actions CI: added NLTK data download step before test run

### Improved
- `preprocess_text()`: switched from stemming to lemmatisation for readable
  LDA topic word lists
- LDA hyperparameters tuned: `min_df=3`, `max_df=0.90`, bigrams enabled

## [1.0.0] - 2026-06-01
### Added
- 1,200 synthetic catering reviews with realistic sentiment distribution
- VADER sentiment scoring with compound threshold classification
- TextBlob subjectivity scoring and polarity cross-check
- LDA topic modeling (5 topics) with CountVectorizer DTM
- Net Sentiment Score (NSS) per caterer — mirrors NPS framework
- Cramér's V association test between sentiment and account tier
- Tableau-ready CSV export
- 17 unit tests
- 6 visualisation charts: sentiment distribution, subjectivity scatter,
  topic word clouds, NSS ranking, trend over time, topic-sentiment heatmap
