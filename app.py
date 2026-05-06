import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-mpnet-base-v2')

st.title("Question Similarity Checker")
st.markdown("### Compare two questions using AI embeddings")

q1 = st.text_input("Enter Question 1")
q2 = st.text_input("Enter Question 2")

if st.button("Check Similarity"):
    if q1 and q2:
        emb1 = model.encode([q1])
        emb2 = model.encode([q2])

        sim = cosine_similarity(emb1, emb2)[0][0]

        st.subheader("Result")
        st.write(f"Similarity Score: {sim:.2f}")
        st.progress(int(sim * 100))

        if sim > 0.75:
            st.success("Highly Similar ✅")
        elif sim > 0.5:
            st.warning("Moderately Similar ⚠️")
        else:
            st.error("Not Similar ❌")
    else:
        st.warning("Please enter both questions")
