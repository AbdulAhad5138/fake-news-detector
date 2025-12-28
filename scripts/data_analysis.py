# ====================
# DATA ANALYSIS VISUALIZATION
# ====================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for pretty graphs
plt.style.use('seaborn-v0_8-darkgrid')

# Load cleaned data
df = pd.read_csv('data/cleaned_news.csv')

print("--- DATA ANALYSIS DASHBOARD ---")
print("=" * 50)

# 1. Basic Info
print(f"Total Articles: {len(df):,}")
print(f"Fake News: {len(df[df['label'] == 'FAKE']):,}")
print(f"Real News: {len(df[df['label'] == 'REAL']):,}")

# 2. Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Fake vs Real Distribution
ax1 = axes[0, 0]
label_counts = df['label'].value_counts()
colors = ['#FF6B6B', '#4ECDC4']  # Red for fake, Teal for real
ax1.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
ax1.set_title('Fake vs Real News Distribution', fontsize=14, fontweight='bold')

# Plot 2: Article Length Distribution
ax2 = axes[0, 1]
df['text_length'] = df['clean_text'].str.len()
for label, color in zip(['FAKE', 'REAL'], colors):
    subset = df[df['label'] == label]
    ax2.hist(subset['text_length'], bins=50, alpha=0.5, label=label, 
             color=color, density=True)
ax2.set_xlabel('Article Length (characters)')
ax2.set_ylabel('Density')
ax2.set_title('Article Length Distribution by Label')
ax2.legend()
ax2.set_xlim(0, 20000)  # Limit to 20k chars for better view

# Plot 3: Subjects by Label
ax3 = axes[1, 0]
subject_counts = df.groupby(['subject', 'label']).size().unstack()
subject_counts = subject_counts.sort_values('FAKE', ascending=False).head(10)
subject_counts.plot(kind='bar', ax=ax3, color=colors)
ax3.set_title('Top 10 Subjects by Label')
ax3.set_xlabel('Subject')
ax3.set_ylabel('Count')
ax3.tick_params(axis='x', rotation=45)

# Plot 4: Word Count Comparison
ax4 = axes[1, 1]
df['word_count'] = df['clean_text'].apply(lambda x: len(str(x).split()))
fake_stats = df[df['label'] == 'FAKE']['word_count'].describe()
real_stats = df[df['label'] == 'REAL']['word_count'].describe()

stats_df = pd.DataFrame({'FAKE': fake_stats, 'REAL': real_stats})
stats_df = stats_df.loc[['mean', 'std', 'min', 'max']]

ax4.axis('off')  # Turn off axis for table
table = ax4.table(cellText=stats_df.round(1).values,
                  rowLabels=stats_df.index,
                  colLabels=stats_df.columns,
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
ax4.set_title('Word Count Statistics', fontsize=14, fontweight='bold', y=0.8)

plt.suptitle('Fake News Dataset Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Save the figure
plt.savefig('data/data_analysis.png', dpi=300, bbox_inches='tight')
print("[OK] Visualization saved as: data/data_analysis.png")

# Show the plot
plt.show()

# 3. Text Analysis
print("\n[INFO] TEXT ANALYSIS:")
print("\nMost common words in FAKE news titles:")
fake_titles = ' '.join(df[df['label'] == 'FAKE']['clean_title'].tolist())
fake_words = pd.Series(fake_titles.split()).value_counts().head(10)
print(fake_words)

print("\nMost common words in REAL news titles:")
real_titles = ' '.join(df[df['label'] == 'REAL']['clean_title'].tolist())
real_words = pd.Series(real_titles.split()).value_counts().head(10)
print(real_words)

print("\n" + "=" * 50)
print("SUCCESS: ANALYSIS COMPLETE!")