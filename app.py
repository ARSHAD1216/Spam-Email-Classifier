import streamlit as st
import pickle
import re

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

# ===============================
# Custom Styling
# ===============================
st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        color: #1E3A8A;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        font-size: 22px;
        text-align: center;
        font-weight: bold;
    }

    .spam {
        background-color: #ffe6e6;
        color: #b30000;
    }

    .not-spam {
        background-color: #e6ffe6;
        color: #006600;
    }
    </style>
""", unsafe_allow_html=True)

# ===============================
# Title Section
# ===============================
st.markdown('<p class="main-title">Spam Email Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine Learning Based Email Spam Detection System</p>', unsafe_allow_html=True)

# ===============================
# Load Model
# ===============================
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ===============================
# Clean Text Function
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# ===============================
# Input Section
# ===============================
st.write("### Enter Email Content")
email_input = st.text_area("", height=200)

# ===============================
# Prediction Button
# ===============================
if st.button("Analyze Email"):

    if email_input.strip() == "":
        st.warning("Please enter an email message.")
    else:
        cleaned = clean_text(email_input)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        confidence = max(probability) * 100

        if prediction == 1:
            st.markdown(
                f'<div class="result-box spam">🚨 This Email is SPAM<br>Confidence: {confidence:.2f}%</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box not-spam">✅ This Email is NOT Spam<br>Confidence: {confidence:.2f}%</div>',
                unsafe_allow_html=True
            )

# ===============================
# Footer
# ===============================
st.markdown("---")
st.markdown(
    "<center> Spam Email Classifier | Built using Machine Learning</center>",
    unsafe_allow_html=True
)