import pandas as pd
import re
from pathlib import Path

# Project root = parent of this script's folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def normalize_for_dedup(text: str) -> str:
    """Aggressive normalization ONLY for detecting duplicates."""
    text = str(text).lower()
    text = re.sub(r'\+?\d[\d\s/-]{6,}\d', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def clean_dataset(in_path, out_path):
    df = pd.read_csv(in_path, encoding="cp1252")
    
    print(f"Loaded {len(df)} rows")
    
    # 1. Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset='message').reset_index(drop=True)
    print(f"Dropped {before - len(df)} exact duplicates")
    
    # 2. Drop near-duplicates (by normalized form)
    before = len(df)
    df['norm'] = df['message'].apply(normalize_for_dedup)
    df = df.drop_duplicates(subset='norm').drop(columns='norm').reset_index(drop=True)
    print(f"Dropped {before - len(df)} near-duplicates")
    
    # 3. Drop very short messages
    before = len(df)
    df = df[df['message'].str.strip().str.len() >= 5].reset_index(drop=True)
    print(f"Dropped {before - len(df)} too-short messages")
    
    # Make sure output dir exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nFinal: {len(df)} rows → {out_path}")
    print(df['label'].value_counts())
    return df

if __name__ == "__main__":
    clean_dataset(
        PROJECT_ROOT / 'data' / 'raw' / 'sms_dataset.csv',
        PROJECT_ROOT / 'data' / 'interim' / 'sms_deduped.csv'
    )