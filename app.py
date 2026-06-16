import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Fake News Analyzer",
    page_icon="📰",
    layout="wide"
)
col1,col2,col3 = st.columns(3)

with col1:
    st.metric("Accuracy","98.91%")

with col2:
    st.metric("Articles","44,058")

with col3:
    st.metric("Features","5,000")

# CSS
st.markdown("""
<style>

/* Main page */
.stApp{
    background-color:white;
}

/* Input area */
[data-testid="stTextArea"] textarea {
    background-color: white !important;
    color: #111827 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 15px !important;
}

/* Placeholder */
[data-testid="stTextArea"] textarea::placeholder {
    color: #666666 !important;
            
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #020617;
}

</style>
""", unsafe_allow_html=True)



# -------------------------
# Load Resources
# -------------------------
nltk.download("stopwords")

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))

# -------------------------
# Text Cleaning Function
# -------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'https?://\S+|www\.\S+',
        '',
        text
    )

    text = re.sub(
        r'<.*?>',
        '',
        text
    )

    text = re.sub(
        r'[%s]' % re.escape(string.punctuation),
        '',
        text
    )

    text = re.sub(
        r'\n',
        ' ',
        text
    )

    text = re.sub(
        r'\w*\d\w*',
        '',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -------------------------
# Header
# -------------------------
st.title("📰 Fake News Analyzer")

st.markdown("""

### Detect Fake News Using AI

Analyze news articles and identify whether
they are Fake or Real using NLP and Machine Learning.
""")

st.caption(
    "NLP-powered Fake News Detection using TF-IDF and Logistic Regression"
)



# -------------------------
# Input Area
# -------------------------
news_text = st.text_area(
    "📰 Paste News Article",
    height=300,
    placeholder="Paste a complete news article here..."
)

# -------------------------
# Prediction
# -------------------------

if st.button("🔍 Analyze News"):

    if not news_text.strip():

        st.warning(
            "Please enter a news article."
        )

    elif len(news_text.split()) < 30:

        st.warning(
            "⚠ Please enter a complete news article (minimum 30 words)."
        )

    else:

        cleaned_text = clean_text(
            news_text
        )

        vectorized_text = vectorizer.transform(
            [cleaned_text]
        )

        prediction = model.predict(
            vectorized_text
        )[0]

        probability = model.predict_proba(
            vectorized_text
        )

        confidence = round(
            max(probability[0]) * 100,
            2
        )

        st.divider()

        if prediction == 0:

            st.error(
                f"🚨 Fake News Detected"
            )

            st.metric(
                label="Confidence",
                value=f"{confidence}%"
            )

        else:

            st.success(
                f"✅ Real News Detected"
            )

            st.metric(
                label="Confidence",
                value=f"{confidence}%"
            )

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown("""
    <h2 style='color:white; margin-bottom:20px;'>
    🚀 Project Status
    </h2>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background:white;
        padding:15px;
        border-radius:12px;
        color:black;
        margin-bottom:15px;
    ">
        🤖 <b>Model:</b> Logistic Regression <br><br>
        📚 <b>Vectorizer:</b> TF-IDF <br><br>
        📰 <b>Dataset:</b> 44,058 Articles <br><br>
        🎯 <b>Accuracy:</b> 98.91%
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background:white;
        padding:15px;
        border-radius:12px;
        color:black;
    ">
        👩‍💻 <b>Khushbu Kumari</b><br><br>
        B.Tech CSE (AI & ML)<br><br>
        Machine Learning Developer
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
### 🔗 Project Links

GitHub:
https://github.com/khushbukumari-dev/fake-news-analyzer

Built with ❤️ by Khushbu Kumari
""")