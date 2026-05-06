import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Page config
st.set_page_config(page_title="AI Similarity App", page_icon="🤖", layout="centered")

# Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Login
st.sidebar.title("🔐 Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

logged_in = False
if username == "admin" and password == "1234":
    logged_in = True
else:
    st.sidebar.warning("Login to unlock features")

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🤖 Question Similarity Checker</h1>", unsafe_allow_html=True)
st.markdown("### Compare two questions using AI embeddings")

# Load model
model = SentenceTransformer('all-mpnet-base-v2')

# Inputs (always visible)
q1 = st.text_area("Enter Question 1")
q2 = st.text_area("Enter Question 2")

# Example button
if st.button("Try Example"):
    q1 = "What is AI?"
    q2 = "Explain artificial intelligence"

# Main action
if st.button("Check Similarity"):
    if not logged_in:
        st.error("Please login first to use this feature")
    else:
        if q1 and q2:
            if len(q1.split()) < 2 or len(q2.split()) < 2:
                st.warning("Please enter meaningful questions")
            else:
                emb1 = model.encode([q1])
                emb2 = model.encode([q2])

                sim = cosine_similarity(emb1, emb2)[0][0]

                st.subheader("Result")
                st.write(f"Similarity Score: {sim:.2f}")
                st.progress(int(sim * 100))

                if sim > 0.85:
                    st.success("Highly Similar (Almost duplicate) ✅")
                elif sim > 0.7:
                    st.info("Similar meaning ℹ️")
                elif sim > 0.5:
                    st.warning("Somewhat related ⚠️")
                else:
                    st.error("Different questions ❌")

                # Save history
                log = pd.DataFrame([[q1, q2, sim]], columns=["Q1","Q2","Score"])
                log.to_csv("history.csv", mode='a', header=False, index=False)

                st.caption("Similarity computed using Sentence-BERT embeddings")
        else:
            st.warning("Please enter both questions")

# Show history
if st.checkbox("Show History"):
    if logged_in:
        try:
            history = pd.read_csv("history.csv")
            st.write(history)
        except:
            st.write("No history yet")
    else:
        st.warning("Login to view history")

# Clear button
if st.button("Clear Inputs"):
    st.rerun()

# Limitations
st.markdown("### Limitations")
st.write("""
- May misclassify partially related questions  
- Sensitive to wording similarity  
- Not fine-tuned on domain-specific data  
""")
