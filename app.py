import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Similarity App", page_icon="🤖", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {
    background-color: #f0f2f6;
}
.title {
    text-align: center;
    color: #4CAF50;
    font-size: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "show_login" not in st.session_state:
    st.session_state.show_login = False

if "show_signup" not in st.session_state:
    st.session_state.show_signup = False

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

# ---------------- TOP BAR ----------------
col1, col2, col3 = st.columns([6, 1, 1])

with col2:
    if st.button("Login"):
        st.session_state.show_login = True
        st.session_state.show_signup = False

with col3:
    if st.button("Sign Up"):
        st.session_state.show_signup = True
        st.session_state.show_login = False

# ---------------- LOGIN FORM ----------------
if st.session_state.show_login:
    st.subheader("🔐 Login")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_pass")

    if st.button("Submit Login"):
        if check_user(email, password):
            st.session_state.logged_in = True
            st.success("Logged in successfully ✅")
        else:
            st.error("Invalid credentials ❌")

# ---------------- SIGNUP FORM ----------------
if st.session_state.show_signup:
    st.subheader("📝 Sign Up")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_pass")

    if st.button("Create Account"):
        if email and password:
            save_user(email, password)
            st.success("Account created! Now login.")
        else:
            st.warning("Enter email & password")

# ---------------- TITLE ----------------
st.markdown('<div class="title">🤖 Question Similarity Checker</div>', unsafe_allow_html=True)
st.markdown("### Compare two questions using AI embeddings")

# ---------------- MODEL ----------------
model = SentenceTransformer('all-mpnet-base-v2')

# ---------------- INPUT ----------------
q1 = st.text_area("Enter Question 1")
q2 = st.text_area("Enter Question 2")

if st.button("Try Example"):
    q1 = "What is AI?"
    q2 = "Explain artificial intelligence"

# ---------------- MAIN FUNCTION ----------------
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

# ---------------- FOOTER ----------------
st.markdown("### Limitations")
st.write("""
- Uses pretrained model (not fine-tuned)
- May misclassify partially related queries
""")
