"""
Rule-based feature extraction — PRUNED to high-signal rules only.

The first attempt had 20 rules, but on a 542-row training set
that caused: (a) feature scaling issues vs TF-IDF, (b) correlated
rules confusing linear models, (c) SVM convergence failures.

We keep ONLY rules that:
  - fire on scam patterns not easily seen by TF-IDF
  - are NOT strongly correlated with each other
  - don't accidentally fire on legitimate Airtel/MoH messages
"""

import re

SUSPICIOUS_PATTERNS = {
    # Miracle money / satanic church scams — very Malawi-specific
    "miracle_money": r"\b(miracle|satanic|satanism|satanizimu|sataniki|mpingo wathu)\b",
    # Transporter / "m'bale wanu" parcel scams
    "agent_scam":    r"\b(agent|transporter|katundu|m'bale wanu|mbale wanu|joni|jubegi|border)\b",
    # Promise of money / prize / reward
    "money_promise": r"\b(mwapata|mwachita mwayi|olandila|opindula|opata|won|prize|mphoto|reward|wina|pambana)\b",    # Lottery / betting scams
    "lottery":       r"\b(premier\s?lotto?|premier\s?bet|lotto|lottery|jackpot|yabooka|fesa)\b",    # Disaster relief / NGO impersonation
    "relief_scam":   r"\b(dodma|unicef|foundation|relief|cyclone|freddy|mtukula)\b",
    # Urgency language (used to pressure victims)
    "urgency":       r"\b(mwachangu|msanga|musachedwe|osachedwa|pasapita|urgent)\b",
    "pin_harvest": r"\b(tumizani|utumize|tumiza|send|share|perekani|patani)\s+.{0,30}\b(pin|password|chinsinsi|chisisi)\b",
    "quiz_scam": r"\b(yankhani ndi mawu|reply with the word|answer with)\b",
}

# Separate numeric (non-boolean) features — small scale so they
# don't dominate TF-IDF terms
_PHONE_NUM_RE = re.compile(r"\b0\d{8,9}\b|\+265\d{7,9}")

# Scale factor for boolean rules so they sit at similar magnitude
# as TF-IDF values (typically 0.05–0.5)
_RULE_SCALE = 0.3

RULE_FEATURE_NAMES = (
    sorted(SUSPICIOUS_PATTERNS.keys())
    + ["phone_count", "length_scaled"]
)


def extract_rule_features(text: str) -> dict:
    """Return a dict of numeric features for one message (raw text)."""
    t = str(text).lower()

    feats = {
        name: int(bool(re.search(pat, t))) * _RULE_SCALE
        for name, pat in SUSPICIOUS_PATTERNS.items()
    }

    phones = _PHONE_NUM_RE.findall(t)
    feats["phone_count"] = min(len(phones), 3) * _RULE_SCALE
    # Length as a small normalized number (0..1), scaled
    feats["length_scaled"] = min(len(t) / 300.0, 1.0) * _RULE_SCALE

    return feats