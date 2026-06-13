"""
utils/sentiment.py
Lightweight sentiment analysis on customer review text using TextBlob.
Falls back to a keyword approach if TextBlob is unavailable.
"""

import re
import numpy as np
import pandas as pd

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


# Simple keyword fallback ─────────────────────────────────────────────────────
_POS = {"great", "excellent", "loved", "perfect", "amazing", "satisfied",
        "recommend", "best", "happy", "wonderful", "fantastic", "superb"}
_NEG = {"disappointed", "terrible", "broke", "unhappy", "worst", "useless",
        "waste", "never", "poor", "bad", "horrible", "awful", "defective"}

def _keyword_polarity(text: str) -> float:
    words = set(re.findall(r"\w+", str(text).lower()))
    pos = len(words & _POS)
    neg = len(words & _NEG)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def analyse_sentiment(texts: pd.Series) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        polarity   : float [-1, 1]
        subjectivity: float [0, 1]   (TextBlob only; else 0.5)
        sentiment_label: Positive / Neutral / Negative
        sentiment_score: scaled 0-1
    """
    results = []
    for text in texts:
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(str(text))
            pol  = blob.sentiment.polarity
            sub  = blob.sentiment.subjectivity
        else:
            pol  = _keyword_polarity(text)
            sub  = 0.5

        if pol > 0.1:
            label = "Positive"
        elif pol < -0.1:
            label = "Negative"
        else:
            label = "Neutral"

        results.append({"polarity": round(pol, 4),
                         "subjectivity": round(sub, 4),
                         "sentiment_label": label,
                         "sentiment_score": round((pol + 1) / 2, 4)})

    return pd.DataFrame(results)


def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds sentiment columns to the dataframe in-place."""
    sent_df = analyse_sentiment(df["review_text"])
    df = pd.concat([df.reset_index(drop=True),
                    sent_df.reset_index(drop=True)], axis=1)
    print(f"[sentiment]  distribution=\n{df['sentiment_label'].value_counts().to_dict()}")
    return df
