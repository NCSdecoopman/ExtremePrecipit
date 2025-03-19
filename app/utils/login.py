import streamlit as st
import bcrypt

# Chargement du hash du mot de passe depuis les secrets
PASSWORD_HASH = st.secrets["auth"]["password_hash"]

def login():
    st.title("Authentification requise")
    pwd = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter"):
        if bcrypt.checkpw(pwd.encode(), PASSWORD_HASH.encode()):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect")