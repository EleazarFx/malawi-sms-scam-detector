"""
Text preprocessing for Malawian SMS scam detection.

We replace high-variance tokens (phone numbers, money amounts, URLs)
with stable placeholder tokens. The model learns patterns like
'call phone to claim' rather than memorizing specific digits.
"""

import re
import string


# Regex patterns compiled once (faster than re-compiling each call)
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+|\b\w+\.(?:ly|com|net|org|mw)\b')
PHONE_PATTERN = re.compile(r'\+?265\s?\d{7,9}|\b0\d{8,9}\b|\+?\d[\d\s/-]{7,}\d')
MONEY_PATTERN = re.compile(
    r'\b(?:mk|mwk|k)\s?[\d,]+(?:\.\d+)?\b',
    flags=re.IGNORECASE
)


def clean_text(text: str) -> str:
    """
    Normalize an SMS message for ML.

    Steps:
      1. Lowercase
      2. Replace URLs, phones, money with stable tokens
      3. Strip punctuation
      4. Collapse whitespace

    Returns an empty string for non-text input.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Important: order matters — phone before money, because
    # 'MK75000' should become 'money', but '0999399300' should be 'phone'
    text = URL_PATTERN.sub(' url ', text)
    text = PHONE_PATTERN.sub(' phone ', text)
    text = MONEY_PATTERN.sub(' money ', text)

    # Remove punctuation (keeps letters, digits, whitespace)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Collapse repeated whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


if __name__ == "__main__":
    # Quick sanity checks — run `python src/preprocess.py` to test
    samples = [
        "Mwapambana K500000! Tumizani PIN yanu ku 0888467774",
        "Balansi K-1499.92; Download My Airtel App here https://bit.ly/20575CG",
        "Moni amayi, ndikubwera kunyumba mawa",
        "MIRACLE MONEY(satanism) anthu 500  imbani 0888467774",
    ]
    for s in samples:
        print(f"IN : {s}")
        print(f"OUT: {clean_text(s)}\n")