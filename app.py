import streamlit as st
import joblib
import re

# Load the saved model & vectorizer
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function to clean email text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return text

# Streamlit App UI
st.set_page_config(page_title="Healthcare Phishing Detector", layout="centered")

st.title("🏥 Healthcare Phishing Email Detection")
st.write("Paste any email text below to check if it's safe or a phishing attempt.")

email_text = st.text_area("✉️ Enter Email Content Here:")

if st.button("Analyze Email"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        cleaned = clean_text(email_text)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == "phishing":
            st.error("🚨 **Phishing Detected!** This email is likely malicious.")
        else:
            st.success("✅ **Safe Email.** No phishing detected.")

st.caption("🔐 AI-powered email security for healthcare systems.")
