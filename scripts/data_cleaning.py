# ====================
# MODULE 1.5: DATA CLEANING
# ====================

import pandas as pd
import re  # Regular expressions for text cleaning
import numpy as np
from datetime import datetime

print("--- STARTING DATA CLEANING PROCESS ---")
print("=" * 50)

# 1. Load our combined data
print("> Loading combined data...")
try:
    df = pd.read_csv('data/all_news.csv')
    print(f"[OK] Loaded {len(df)} articles")
except:
    print("[ERROR] Run data_preparation.py first!")
    exit()

# 2. Let's see what we're working with
print("\n> INITIAL DATA INSPECTION:")
print(f"Data shape: {df.shape}")  # (rows, columns)
print(f"\nColumns: {list(df.columns)}")
print(f"\nMissing values per column:")
print(df.isnull().sum())

# 3. Check data types
print("[INFO] Data types:")
print(df.dtypes)

# 4. Handle missing values
print("\n> HANDLING MISSING VALUES...")

# Count missing values before cleaning
missing_before = df.isnull().sum().sum()
print(f"Total missing values before: {missing_before}")

# Fill missing text with empty string
df['text'] = df['text'].fillna('')
df['title'] = df['title'].fillna('')

# For subject, fill with 'Unknown'
df['subject'] = df['subject'].fillna('Unknown')

# For date, we'll handle specially
df['date'] = df['date'].fillna('Unknown')

missing_after = df.isnull().sum().sum()
print(f"Total missing values after: {missing_after}")

# 5. TEXT CLEANING FUNCTION
print("\n> CLEANING TEXT DATA...")

def clean_text(text):
    """
    Clean news article text by:
    1. Converting to lowercase
    2. Removing special characters
    3. Removing extra spaces
    4. Removing URLs
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters but keep letters, numbers, and basic punctuation
    text = re.sub(r'[^\w\s.,!?]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply cleaning to title and text columns
print("Cleaning titles...")
df['clean_title'] = df['title'].apply(clean_text)

print("Cleaning article text...")
df['clean_text'] = df['text'].apply(clean_text)

# 6. DATE CLEANING
print("\n> CLEANING DATE COLUMN...")

def clean_date(date_str):
    """
    Try to parse various date formats
    """
    if not isinstance(date_str, str) or date_str == 'Unknown':
        return 'Unknown'
    
    # Remove any HTML tags or weird characters
    date_str = str(date_str).split()[0]  # Take first part only
    
    # Common date patterns in the dataset
    patterns = [
        '%Y-%m-%d',  # 2020-01-15
        '%d-%m-%Y',  # 15-01-2020
        '%m/%d/%Y',  # 01/15/2020
        '%B %d, %Y', # January 15, 2020
        '%b %d, %Y', # Jan 15, 2020
    ]
    
    for pattern in patterns:
        try:
            return datetime.strptime(date_str, pattern).strftime('%Y-%m-%d')
        except:
            continue
    
    return 'Unknown'

df['clean_date'] = df['date'].apply(clean_date)

# 7. Check cleaning results
print("\n[OK] CLEANING COMPLETED!")
print("\n[INFO] SAMPLE RESULTS:")

print("\nBefore cleaning (title):")
print(df['title'].iloc[0][:100] + "...")

print("\nAfter cleaning (clean_title):")
print(df['clean_title'].iloc[0][:100] + "...")

print("\n" + "=" * 50)
print("\n> STATISTICS AFTER CLEANING:")

# Count empty articles after cleaning
empty_articles = df[df['clean_text'].str.len() < 50].shape[0]
print(f"Articles with very short text (<50 chars): {empty_articles}")

# Average article length
df['text_length'] = df['clean_text'].str.len()
print(f"\nAverage article length: {df['text_length'].mean():.0f} characters")
print(f"Shortest article: {df['text_length'].min()} characters")
print(f"Longest article: {df['text_length'].max()} characters")

# Distribution by label
print(f"\n[INFO] DISTRIBUTION BY LABEL:")
print(df['label'].value_counts())

# 8. Save cleaned data
print("\n> SAVING CLEANED DATA...")
df.to_csv('data/cleaned_news.csv', index=False)
print("[OK] Saved to: data/cleaned_news.csv")

# 9. Save a smaller sample for testing (optional)
print("\n> Creating sample dataset for testing...")
sample_df = df.sample(n=1000, random_state=42)  # Random 1000 articles
sample_df.to_csv('data/sample_news.csv', index=False)
print("[OK] Saved sample to: data/sample_news.csv")

print("\n" + "=" * 50)
print("SUCCESS: DATA CLEANING COMPLETED SUCCESSFULLY!")
print("\nNext steps:")
print("1. [OK] Data loaded and combined")
print("2. [OK] Missing values handled")
print("3. [OK] Text cleaned and standardized")
print("4. [OK] Dates formatted")
print("5. [OK] Saved cleaned dataset")
