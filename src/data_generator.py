"""
data_generator.py
=================
Generates a realistic synthetic dataset of catering marketplace reviews.

Review structure mirrors what a platform like ezCater would actually collect:
  - Star rating (1–5)
  - Free-text review body
  - Caterer and order metadata (for joining to operational data)
  - Review date

The synthetic text is built from templated sentence pools combined with
caterer-specific characteristics (cuisine type, performance tier).  This
approach produces varied, natural-sounding reviews without requiring real
user data — appropriate for a portfolio dataset.

Volume: 1,200 reviews across 15 caterers, Jan 2023 – Dec 2024.
"""

import random
import numpy as np
import pandas as pd
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 7
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Caterer definitions — matches the SQL project schema
# ---------------------------------------------------------------------------
CATERERS = [
    {"id": 1,  "name": "Boston Box Lunch Co.",   "cuisine": "American",       "tier": "platinum", "base_rating": 4.7},
    {"id": 2,  "name": "Spice Route Catering",   "cuisine": "Indian",         "tier": "gold",     "base_rating": 4.5},
    {"id": 3,  "name": "Harbor Fresh Kitchen",   "cuisine": "American",       "tier": "silver",   "base_rating": 4.3},
    {"id": 4,  "name": "Bella Cucina Events",    "cuisine": "Italian",        "tier": "platinum", "base_rating": 4.8},
    {"id": 5,  "name": "Green Garden Catering",  "cuisine": "Vegetarian",     "tier": "gold",     "base_rating": 4.6},
    {"id": 6,  "name": "Taqueria Del Sol",        "cuisine": "Mexican",       "tier": "gold",     "base_rating": 4.4},
    {"id": 7,  "name": "The Grain Bowl",          "cuisine": "American",      "tier": "silver",   "base_rating": 4.2},
    {"id": 8,  "name": "Mediterranean Table",    "cuisine": "Mediterranean",  "tier": "platinum", "base_rating": 4.7},
    {"id": 9,  "name": "Seoul Kitchen Catering", "cuisine": "Korean",         "tier": "gold",     "base_rating": 4.5},
    {"id": 10, "name": "New England Deli",        "cuisine": "American",      "tier": "needs_improvement", "base_rating": 3.9},
    {"id": 11, "name": "Saffron House",           "cuisine": "Middle Eastern","tier": "gold",     "base_rating": 4.6},
    {"id": 12, "name": "Pacific Rim Catering",   "cuisine": "Asian Fusion",   "tier": "silver",   "base_rating": 4.3},
    {"id": 13, "name": "The Smokehouse BBQ",     "cuisine": "BBQ",            "tier": "silver",   "base_rating": 4.1},
    {"id": 14, "name": "Garden State Greens",    "cuisine": "Vegetarian",     "tier": "silver",   "base_rating": 4.4},
    {"id": 15, "name": "Federal Street Fare",    "cuisine": "American",       "tier": "needs_improvement", "base_rating": 3.8},
]

# ---------------------------------------------------------------------------
# Review sentence pools — grouped by theme (topics for LDA to discover)
# ---------------------------------------------------------------------------

# Positive sentences by topic
POS_FOOD_QUALITY = [
    "The food was absolutely delicious.",
    "Everything tasted fresh and flavorful.",
    "The quality of the ingredients was outstanding.",
    "Our team raved about the food all afternoon.",
    "Portions were generous and well-presented.",
    "Best catering we have had in the office.",
    "The food exceeded our expectations.",
    "Incredibly fresh — you could taste the quality.",
    "A perfect variety of options for all dietary needs.",
    "The flavors were bold and authentic.",
]
POS_DELIVERY = [
    "Arrived right on time, which is rare.",
    "Delivery was prompt and professional.",
    "The driver was courteous and helpful with setup.",
    "Showed up early and had everything ready before our guests arrived.",
    "The delivery team handled a difficult parking situation gracefully.",
    "On-time delivery made our event stress-free.",
    "Hot food arrived hot — impressive logistics.",
]
POS_PACKAGING = [
    "Packaging was neat and easy to serve.",
    "Everything was labeled clearly, including allergen info.",
    "The presentation was beautiful — looked catered, not just delivered.",
    "Eco-friendly containers were a nice touch.",
    "Each item was packaged individually, which our team appreciated.",
]
POS_VALUE = [
    "Great value for the price.",
    "Competitive pricing without sacrificing quality.",
    "Worth every penny for a team of 50.",
    "Affordable and filling — will order again.",
]
POS_REPEAT = [
    "We will definitely be ordering again.",
    "Already booked our next order.",
    "Our go-to caterer for all company events.",
    "Highly recommend to other offices in the area.",
    "We have used them three times and they get better every time.",
]

# Negative sentences by topic
NEG_DELIVERY = [
    "Terrible experience — the order arrived over 30 minutes late and nobody warned us.",
    "Completely unacceptable delivery delay ruined our client lunch.",
    "Driver was lost and unhelpful, making an already bad situation worse.",
    "We were told noon, food arrived at 12:50. Embarrassing in front of clients.",
    "The tardiness was inexcusable and disrupted our entire meeting schedule.",
    "Awful logistics — late delivery left food cold and sitting out far too long.",
    "I was frustrated and disappointed by how poorly the delivery was managed.",
]
NEG_FOOD_QUALITY = [
    "Disgusting — the food was cold, soggy, and clearly not fresh.",
    "Several items were completely missing from the order. Unacceptable.",
    "The salad was wilted and inedible. Will not order again.",
    "Tiny portions for the price — our team was still hungry after the meal.",
    "The food was bland, tasteless, and frankly not worth the money.",
    "Undercooked chicken is not just bad quality — it is a health risk.",
    "Terrible quality overall. The food looked nothing like the menu photos.",
    "Worst catering experience we have had. The food was bad across the board.",
]
NEG_PACKAGING = [
    "Awful packaging — leaking containers made a mess of our conference table.",
    "Completely unlabeled containers — unacceptable for a team with food allergies.",
    "The soup arrived in an unsuitable container and spilled everywhere.",
    "Sloppy packing with loose lids meant several items spilled in transit.",
]
NEG_CUSTOMER_SERVICE = [
    "Horrible customer service — no one responded when we tried to modify the order.",
    "Called twice, emailed once. Zero response. Completely ignored us.",
    "The confirmation never arrived and no one seemed to care.",
    "They substituted items without telling us, including one with a nut allergen.",
]
NEG_VALUE = [
    "Shockingly overpriced for the poor quality we received.",
    "Terrible value — not worth anywhere near what we paid.",
    "Expected far more for this price point. Very disappointed.",
]

# Neutral/mixed sentences
MIXED_SENTENCES = [
    "Food was decent but nothing exceptional.",
    "Delivery was on time but setup took longer than expected.",
    "Good variety but portions were smaller than we needed.",
    "Would order again, though we hope they fix the labeling issue.",
    "Hit or miss — some items were great, others fell flat.",
]


def _build_review_text(rating: int, caterer: dict) -> str:
    """
    Construct a review body by sampling from positive/negative sentence pools
    based on the star rating.  Higher-rated caterers draw more from positive
    pools; lower-rated from negative pools.

    The sentence composition is designed so that LDA can discover
    distinct topics (food quality, delivery, packaging, value).
    """
    sentences = []

    if rating >= 4:
        # Positive review: lead with food or delivery, add 1-2 more positives
        lead_pools = [POS_FOOD_QUALITY, POS_DELIVERY]
        sentences.append(random.choice(random.choice(lead_pools)))
        sentences.append(random.choice(POS_FOOD_QUALITY + POS_DELIVERY + POS_PACKAGING))
        if rating == 5:
            sentences.append(random.choice(POS_VALUE + POS_REPEAT))
    elif rating == 3:
        # Mixed review
        sentences.append(random.choice(MIXED_SENTENCES))
        sentences.append(random.choice(POS_FOOD_QUALITY + POS_DELIVERY))
        sentences.append(random.choice(NEG_DELIVERY + NEG_PACKAGING))
    else:
        # Negative review: lead with a specific complaint
        if rating == 1:
            neg_lead = [NEG_DELIVERY, NEG_FOOD_QUALITY, NEG_CUSTOMER_SERVICE]
        else:
            neg_lead = [NEG_DELIVERY, NEG_FOOD_QUALITY]
        sentences.append(random.choice(random.choice(neg_lead)))
        sentences.append(random.choice(NEG_FOOD_QUALITY + NEG_DELIVERY + NEG_PACKAGING))
        if caterer["tier"] == "needs_improvement":
            sentences.append(random.choice(NEG_VALUE + NEG_CUSTOMER_SERVICE))

    return " ".join(sentences)


def _sample_rating(caterer: dict) -> int:
    """
    Sample a star rating from a distribution shaped by the caterer's base rating.
    Platinum caterers skew heavily 4-5 star; needs_improvement skew toward 2-3 star.
    """
    base = caterer["base_rating"]
    # Build a probability distribution over 1-5 stars centred on base_rating
    raw = np.array([
        max(0, 1 - abs(r - base) * 1.2)   # triangular-ish distribution
        for r in [1, 2, 3, 4, 5]
    ])
    probs = raw / raw.sum()
    return int(np.random.choice([1, 2, 3, 4, 5], p=probs))


def generate_reviews(n_reviews: int = 1200) -> pd.DataFrame:
    """
    Generate the full review dataset.

    Parameters
    ----------
    n_reviews : total number of reviews to generate

    Returns
    -------
    pd.DataFrame with columns:
        review_id, caterer_id, caterer_name, cuisine_type, performance_tier,
        review_date, star_rating, review_text, word_count
    """
    # Distribute reviews proportionally — higher-rated caterers get more reviews
    weights = np.array([c["base_rating"] ** 2 for c in CATERERS])
    weights = weights / weights.sum()
    review_counts = np.round(weights * n_reviews).astype(int)
    # Adjust for rounding to hit exactly n_reviews
    review_counts[-1] += n_reviews - review_counts.sum()

    start_date = date(2023, 1, 1)
    end_date   = date(2024, 12, 31)
    date_range_days = (end_date - start_date).days

    records = []
    review_id = 1

    for caterer, count in zip(CATERERS, review_counts):
        for _ in range(count):
            rating      = _sample_rating(caterer)
            review_date = start_date + timedelta(days=random.randint(0, date_range_days))
            text        = _build_review_text(rating, caterer)

            records.append({
                "review_id":        review_id,
                "caterer_id":       caterer["id"],
                "caterer_name":     caterer["name"],
                "cuisine_type":     caterer["cuisine"],
                "performance_tier": caterer["tier"],
                "review_date":      review_date,
                "star_rating":      rating,
                "review_text":      text,
                "word_count":       len(text.split()),
            })
            review_id += 1

    df = pd.DataFrame(records).sort_values("review_date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_reviews()
    df.to_csv("data/reviews.csv", index=False)
    print(f"Generated {len(df):,} reviews")
    print(df.groupby("performance_tier")["star_rating"].mean().round(2))
