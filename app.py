import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("Question Similarity Checker")

q1 = st.text_input("Enter Question 1")
q2 = st.text_input("Enter Question 2")

if st.button("Check"):
    if q1 and q2:
        emb1 = model.encode([q1])
        emb2 = model.encode([q2])

        sim = cosine_similarity(emb1, emb2)[0][0]

        st.write(f"Similarity Score: {sim:.2f}")

        if sim > 0.6:
            st.success("Questions are Similar")
        else:
            st.error("Questions are NOT Similar")
    else:
        st.warning("Please enter both questions")
