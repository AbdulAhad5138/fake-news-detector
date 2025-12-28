# ====================
# MODULE 1: DATA PREPARATION
# ====================

# 1. Importing our "tools"
import pandas as pd  # For handling data tables
import os  # For working with files/folders

# 2. Print welcome message
print("--- Starting Fake News Detection Project! ---")
print("=" * 50)

# 3. Check if our data files exist
print("> Checking for data files...")
if not os.path.exists('data/Fake.csv'):
    print("[ERROR] Fake.csv not found in data/ folder!")
    print("Please download from Kaggle and place in data/ folder")
    exit()
else:
    print("[OK] Found Fake.csv")

if not os.path.exists('data/True.csv'):
    print("[ERROR] True.csv not found in data/ folder!")
    exit()
else:
    print("[OK] Found True.csv")

# 4. Load the data (like opening Excel files)
print("\n> Loading data files...")
fake_news = pd.read_csv('data/Fake.csv')
true_news = pd.read_csv('data/True.csv')

# 5. Let's see what we got
print("\n[INFO] FAKE NEWS DATASET:")
print(f"Number of articles: {len(fake_news)}")
print(f"Columns available: {list(fake_news.columns)}")
print("\nFirst 3 fake articles:")
print(fake_news.head(3))

print("\n" + "=" * 50)

print("\n[INFO] TRUE NEWS DATASET:")
print(f"Number of articles: {len(true_news)}")
print(f"Columns available: {list(true_news.columns)}")
print("\nFirst 3 true articles:")
print(true_news.head(3))

# 6. Add labels to distinguish fake vs real
print("\n> Adding labels...")
fake_news['label'] = 'FAKE'  # Add new column called 'label' with value 'FAKE'
true_news['label'] = 'REAL'   # Add new column called 'label' with value 'REAL'

# 7. Combine both datasets
print("\n> Combining datasets...")
all_news = pd.concat([fake_news, true_news], ignore_index=True)

# 8. Check the combined data
print("\n[INFO] COMBINED DATASET:")
print(f"Total articles: {len(all_news)}")
print(f"Fake articles: {len(all_news[all_news['label'] == 'FAKE'])}")
print(f"Real articles: {len(all_news[all_news['label'] == 'REAL'])}")

# 9. Save the combined data
print("\n> Saving combined data...")
all_news.to_csv('data/all_news.csv', index=False)
print("[OK] Saved to: data/all_news.csv")

print("\n" + "=" * 50)
print("SUCCESS: MODULE 1 COMPLETED SUCCESSFULLY!")
# Let's explore more
print("\n> EXTRA EXPLORATION:")
print("\nSubjects in fake news:")
print(fake_news['subject'].value_counts())

print("\nSubjects in true news:")
print(true_news['subject'].value_counts())

# Check date formats
print("\n[INFO] Sample dates:")
print(fake_news['date'].head(5))
print("Next: We'll clean and prepare the text data")