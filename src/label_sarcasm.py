import pandas as pd

df = pd.read_csv("dataset/cleaned_reviews.csv")

# 低評価 (1, 2) を皮肉と仮定
df['sarcasm'] = df['Score'].apply(lambda x: 1 if x <= 2 else 0)

df.to_csv("dataset/labeled_reviews.csv", index=False)
print("✅ Sarcasm Labels Created and Saved!")
