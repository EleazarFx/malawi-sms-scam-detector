"""
Feature extraction combining:
  - TF-IDF over cleaned text (word patterns)
  - Handcrafted rule features (structural scam signals)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack, csr_matrix
from rules import extract_rule_features, RULE_FEATURE_NAMES


# ---- Stopwords ----------------------------------------------------

CHICHEWA_STOPWORDS = {
    # pronouns / demonstratives
    "ndi", "ndine", "iye", "ife", "inu", "iwo",
    "iyi", "iyo", "ili", "uwo", "uwu", "izi",
    "zimene", "zomwe", "ameneyo", "meneyu",
    # conjunctions / particles
    "ndipo", "koma", "kapena", "chifukwa", "kuti", "ngati",
    "pamene", "kaya", "ayi", "inde",
    # locatives / prepositions
    "ku", "kwa", "pa", "mu", "mwa", "kuchokera", "kuchokela", "mpaka",
    # possessives
    "wa", "ya", "la", "cha", "za", "a",
    "yawo", "wathu", "wanu", "wanga", "wako", "wake",
    # greetings
    "moni", "zikomo", "pepani", "chonde",
}

PRESERVE_WORDS = {"no", "not", "you", "your", "we", "us", "free", "now", "call"}

ALL_STOPWORDS = (CHICHEWA_STOPWORDS | set(ENGLISH_STOP_WORDS)) - PRESERVE_WORDS


# ---- Vectorizers --------------------------------------------------

def build_vectorizer() -> TfidfVectorizer:
    """TF-IDF over cleaned text with Malawian stopwords removed."""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words=list(ALL_STOPWORDS),
        lowercase=False,   # already lowercased upstream
    )


def _rule_transform(texts):
    """Turn a list of raw texts into a numeric matrix of rule features."""
    rows = []
    for t in texts:
        feats = extract_rule_features(t)
        rows.append([feats[k] for k in RULE_FEATURE_NAMES])
    return np.asarray(rows, dtype=float)


def rule_vectorizer() -> FunctionTransformer:
    """Wrap extract_rule_features as a sklearn transformer."""
    return FunctionTransformer(_rule_transform, validate=False)


def build_feature_union() -> FeatureUnion:
    """Combine TF-IDF (on cleaned text) with rule features (on raw text)."""
    return FeatureUnion([
        ("tfidf", build_vectorizer()),
        ("rules", rule_vectorizer()),
    ])

# ---- Custom transformer: routes each column to the right sub-transformer ----

class TextRouter(BaseEstimator, TransformerMixin):
    """
    Accepts a DataFrame with columns ['clean', 'raw'] and
    applies different transformers to each, then concatenates.

    IMPORTANT: This class must live in a real module (not __main__)
    so that joblib can pickle/unpickle it correctly across processes.
    """

    def __init__(self):
        self.tfidf = build_vectorizer()
        self.rules = rule_vectorizer()

    def fit(self, X, y=None):
        self.tfidf.fit(X["clean"], y)
        self.rules.fit(X["raw"], y)
        return self

    def transform(self, X):
        tfidf_X = self.tfidf.transform(X["clean"])
        rules_X = csr_matrix(self.rules.transform(X["raw"]))
        return hstack([tfidf_X, rules_X]).tocsr()