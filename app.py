import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load better model
model = SentenceTransformer('all-mpnet-base-v2')

st.title("Question Similarity Checker")
st.markdown("### Compare two questions using AI embeddings")

# Example button
if st.button("Try Example"):
    q1 = "What is AI?"
    q2 = "Explain artificial intelligence"
else:
    q1 = st.text_input("Enter Question 1")
    q2 = st.text_input("Enter Question 2")

if st.button("Check Similarity"):
    if q1 and q2:
        
        # Input validation
        if len(q1.split()) < 2 or len(q2.split()) < 2:
            st.warning("Please enter meaningful questions")
        else:
            emb1 = model.encode([q1])
            emb2 = model.encode([q2])

            sim = cosine_similarity(emb1, emb2)[0][0]

            # Output
            st.subheader("Result")
            st.write(f"Similarity Score: {sim:.2f}")
            st.progress(int(sim * 100))

            # Interpretation
            if sim > 0.85:
                st.success("Highly Similar (Almost duplicate) ✅")
            elif sim > 0.7:
                st.info("Similar meaning ℹ️")
            elif sim > 0.5:
                st.warning("Somewhat related ⚠️")
            else:
                st.error("Different questions ❌")

            # Save results
            log = pd.DataFrame([[q1, q2, sim]], columns=["Q1","Q2","Score"])
            log.to_csv("history.csv", mode='a', header=False, index=False)

            # Info
            st.caption("Similarity computed using Sentence-BERT embeddings")

    else:
        st.warning("Please enter both questions")

# Limitations section
st.markdown("### Limitations")
st.write("""
- May misclassify partially related questions  
- Sensitive to wording similarity  
- Not fine-tuned on domain-specific data  
""")
