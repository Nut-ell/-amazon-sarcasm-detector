import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import os

# モデル保存用フォルダを作成（なければ作る）

os.makedirs("model", exist_ok=True)


# データを読み込む
df = pd.read_csv("dataset/labeled_reviews.csv")

# NaN を削除
df = df.dropna(subset=['cleaned_text'])

# TF-IDF ベクトル化
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['sarcasm']

# 訓練 & テストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ナイーブベイズで学習
model = MultinomialNB()
model.fit(X_train, y_train)

# 精度を確認
y_pred = model.predict(X_test)
print(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# モデルを保存
joblib.dump(model, "model/sarcasm_detector.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("✅ Model Saved!")
