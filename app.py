import streamlit as st
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Load files
model = pickle.load(open("model.pkl", "rb"))
vect = pickle.load(open("vect.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

lem = WordNetLemmatizer()
stop_words = stopwords.words("english")

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

st.title("AI Resume Classifier")

resume = st.text_area("Paste Resume Here")

if st.button("Predict Category"):

    if resume.strip() == "":
        st.warning("Please paste a resume before clicking Predict Category.")
    else:
        clean_resume = cleaned_text(resume)

        if len(clean_resume.split()) < 30:
         st.warning("Resume content is too short.")
         st.stop()

        vector = vect.transform([clean_resume])

        if vector.nnz == 0:
            st.error("No recognizable resume keywords found. Please provide a more detailed resume.")
            st.stop()

        prediction = model.predict(vector)

        category = le.inverse_transform(prediction)

        st.success(f"Predicted Category: {category[0]}")
