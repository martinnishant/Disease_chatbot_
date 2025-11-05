import streamlit as st
import requests

st.set_page_config(page_title="AI Doctor Chatbot 🩺", page_icon="💬")
st.title("AI Doctor Chatbot 🩺")
st.write("Enter your symptoms below to get an AI-generated health summary.")

symptoms_input = st.text_input("Symptoms (comma-separated):", "fatigue, cough, shortness of breath")

if st.button("Analyze"):
    payload = {
        "symptoms": [s.strip() for s in symptoms_input.split(",")],
        "top_n": 3
    }
    response = requests.post("http://127.0.0.1:8000/chat_diagnose", json=payload)
    if response.status_code == 200:
        data = response.json()["diagnosis"]
        st.subheader("🩺 Summary")
        st.write(data["summary"])
        st.subheader("⚠️ Common Causes")
        st.write(data["common_causes"])
        st.subheader("💊 Treatment & Care")
        st.write(data["treatment_and_care"])
        st.subheader("📞 When to See a Doctor")
        st.write(data["when_to_see_doctor"])
    else:
        st.error("Error fetching diagnosis.")
