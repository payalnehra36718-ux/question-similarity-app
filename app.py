import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Similarity App", page_icon="🤖")

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- USER FILE ----------------
USER_FILE = "users.csv"

if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["email", "password"]).to_csv(USER_FILE, index=False)

def load_users():
    return pd.read_csv(USER_FILE)

def save_user(email, password):
    df = load_users()
    df.loc[len(df)] = [email, password]
    df.to_csv(USER_FILE, index=False)

def check_user(email, password):
    df = load_users()
    return ((df["email"] == email) & (df["password"] == password)).any()

# ---------------- SIDEBAR AUTH ----------------
st.sidebar.title("🔐 Authentication")

menu = st.sidebar.selectbox("Choose", ["Login", "Sign Up"])

email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

if menu == "Sign Up":
    if st.sidebar.button("Create Account"):
        if email and password:
            save_user(email, password)
            st.sidebar.success("Account created! Now login.")
        else:
            st.sidebar.warning("Enter email & password")

elif menu == "Login":
    if st.sidebar.button("Login"):
        if check_user(email, password):
            st.session_state.logged_in = True
            st.sidebar.success("Logged in successfully ✅")
        else:
            st.sidebar.error("Invalid credentials ❌")

# ---------------- UI ----------------
st.title("🤖 Question Similarity Checker")
st.markdown("Compare two questions using AI embeddings")

# Load model
model = SentenceTransformer('all-mpnet-base-v2')

# Inputs
q1 = st.text_area("Enter Question 1")
q2 = st.text_area("Enter Question 2")

# Example
if st.button("Try Example"):
    q1 = "What is AI?"
    q2 = "Explain artificial intelligence"

# Main logic
if st.button("Check Similarity"):
    if not st.session_state.logged_in:
        st.error("Please login first 🔒")
    else:
        if q1 and q2:
            emb1 = model.encode([q1])
            emb2 = model.encode([q2])

            sim = cosine_similarity(emb1, emb2)[0][0]

            st.subheader("Result")
            st.write(f"Similarity Score: {sim:.2f}")
            st.progress(int(sim * 100))

            if sim > 0.85:
                st.success("Highly Similar ✅")
            elif sim > 0.7:
                st.info("Similar ℹ️")
            elif sim > 0.5:
                st.warning("Somewhat related ⚠️")
            else:
                st.error("Not Similar ❌")
        else:
            st.warning("Enter both questions")

# ---------------- LIMITATIONS ----------------
st.markdown("### Limitations")
st.write("""
- Uses pretrained model (not fine-tuned)
- May misclassify partially related queries
""")
