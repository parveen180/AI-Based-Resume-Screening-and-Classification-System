import streamlit as st
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load trained model files
model = pickle.load(open("model.pkl", "rb"))
vect = pickle.load(open("vect.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# NLP setup
lem = WordNetLemmatizer()
stop_words = stopwords.words("english")

# Text Cleaning Function
def cleaned_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text)

    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lem.lemmatize(word) for word in words]

    return " ".join(words)

# ATS Score using TF-IDF + Cosine Similarity
def calculate_ats_score(resume_text, job_description):

    resume_text = cleaned_text(resume_text)
    job_description = cleaned_text(job_description)

    documents = [resume_text, job_description]

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )

    score = int(similarity[0][0] * 100)

    return score

# Streamlit UI
st.title("AI-Based Resume Screening and Classification System")

resume = st.text_area("Paste Resume Here")

job_description = st.text_area("Paste Job Description Here")

if st.button("Predict Job Category & Analyze Resume"):

    if resume.strip() == "":
        st.warning("Please paste a resume.")
        st.stop()

    if job_description.strip() == "":
        st.warning("Please paste a job description.")
        st.stop()

    # Resume Classification
    clean_resume = cleaned_text(resume)

    vector = vect.transform([clean_resume])

    prediction = model.predict(vector)

    category = le.inverse_transform(prediction)

    st.subheader("Predicted Job Category")
    st.success(category[0])

    # ATS Score
    ats_score = calculate_ats_score(
        resume,
        job_description
    )

    st.subheader("ATS Compatibility Analysis")
    st.write(f"ATS Score: {ats_score}%")

    if ats_score >= 70:
        st.success("ATS Friendly ✅")

    elif ats_score >= 40:
        st.warning("Moderately ATS Friendly ⚠️")

    else:
        st.error("Not ATS Friendly ❌")