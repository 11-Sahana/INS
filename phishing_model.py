import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1️⃣ Load dataset
df = pd.read_csv("emails.csv")

df = pd.read_csv("emails.csv")
print(df.head())
print(df.columns)
print(len(df))



# ✅ Remove empty rows (NaN values)
df = df.dropna()

# ✅ Optional: Check if labels are only 'phishing' or 'safe'
print(df['label'].unique())


# 2️⃣ Clean text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub(r"[^a-zA-Z]", " ", text)  # keep only letters
    text = text.lower()
    return text

df['text'] = df['text'].apply(clean_text)

# 3️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 4️⃣ Convert text → numbers
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5️⃣ Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 6️⃣ Test accuracy
pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))

# 7️⃣ Save model for later use
import joblib
joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
