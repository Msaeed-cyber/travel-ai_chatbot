import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv("travel_faq.csv")

# Initialize TF-IDF
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
tfidf_matrix = vectorizer.fit_transform(df['question'])

# Function to get the best response
def get_response(user_query):
    user_tfidf = vectorizer.transform([user_query])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)
    best_match_idx = similarities.argmax()
    return df.iloc[best_match_idx]['answer']

# Streamlit UI
st.set_page_config(page_title="Travel Q&A Chatbot", page_icon="✈️")

st.title("🤖 Travel Q&A Chatbot")
st.write("Ask me anything about travel! 🌍")

user_input = st.text_input("Your Question:")

if st.button("Get Answer"):
    if user_input.strip():
        response = get_response(user_input)
        st.success(f"**Answer:** {response}")
    else:
        st.warning("Please enter a question.")

st.markdown("---")
st.caption("Built with 💙 using Streamlit, scikit-learn, and NLP.")
