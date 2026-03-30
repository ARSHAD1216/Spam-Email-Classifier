import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# 1. Load dataset (PARQUET file)
# ===============================

df = pd.read_parquet("train.parquet")

print("Dataset loaded successfully")
print("Total rows:", len(df))
print("Columns:", df.columns)

# ===============================
# 2. Use only required columns
# ===============================

df = df[["text", "label"]]
df = df.dropna()

# ===============================
# 3. Clean text
# ===============================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

df["text"] = df["text"].apply(clean_text)

# ===============================
# 4. Train-test split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# ===============================
# 5. Convert text to numbers (TF-IDF)
# ===============================

vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===============================
# 6. Train model
# ===============================

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ===============================
# 7. Accuracy
# ===============================

y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 8. Save model
# ===============================

pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel saved successfully!")