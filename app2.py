import streamlit as st
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Load trained files
model = pickle.load(open("model.pkl", "rb"))
vect = pickle.load(open("vect.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# NLTK setup
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
    words = [w for w in words if w not in stop_words]
    words = [lem.lemmatize(w) for w in words]

    return " ".join(words)


# ATS Score Function
def calculate_ats_score(resume_text):

    score = 0

    # Email
    if re.search(r'[\w\.-]+@[\w\.-]+\.\w+', resume_text):
        score += 20

    # Phone
    if re.search(r'\b\d{10}\b', resume_text):
        score += 20

    # Education
    if any(word in resume_text.lower() for word in
           ['bachelor', 'master', 'degree', 'university', 'college']):
        score += 20

    # Experience
    if any(word in resume_text.lower() for word in
           ['experience', 'worked', 'project', 'internship']):
        score += 20

    # Skills
    if any(word in resume_text.lower() for word in
           ['java', 'python', 'sql', 'spring', 'machine learning']):
        score += 20

    return score


# Streamlit UI
st.title("AI Resume Screening and ATS Evaluation System")

resume = st.text_area("Paste Resume Here")

job_description = st.text_area("Paste Job Description Here")

if st.button("Predict Job Category & Analyze Resume"):

    if resume.strip() == "":
        st.warning("Please paste a resume.")
        st.stop()

    if job_description.strip() == "":
        st.warning("Please paste a job description.")
        st.stop()

    clean_resume = cleaned_text(resume)

    vector = vect.transform([clean_resume])

    prediction = model.predict(vector)

    category = le.inverse_transform(prediction)

    st.subheader("Predicted Job Category")
    st.success(category[0])

    # ATS Score
    ats_score = calculate_ats_score(resume)

    st.subheader("ATS Compatibility Analysis")
    st.write(f"{ats_score}%")

    ats_score = calculate_ats_score(resume)

    if ats_score >= 80:
     st.success("ATS Friendly ✅")
    else:
     st.error("Needs Improvement ❌")