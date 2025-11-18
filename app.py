import streamlit as st
import tensorflow as tf
import pickle
import json
import os
import random
import requests
import time
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.sequence import pad_sequences

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

model = tf.keras.models.load_model("best_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("chatbot_dataset.json", "r") as f:
    dataset = json.load(f)

intent_responses = {i["tag"]: i["responses"] for i in dataset["intents"]}

max_len = 6


def predict_intent(text):
    seq = tokenizer.texts_to_sequences([text.lower()])
    seq = pad_sequences(seq, maxlen=max_len, padding="post")
    pred = model.predict(seq)[0]
    idx = pred.argmax()
    intent = label_encoder.inverse_transform([idx])[0]
    confidence = float(pred[idx])
    return intent, confidence


def deepseek_generate(query, history):
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant integrated inside a lightweight chatbot. "
                "Your responses must be short, natural, friendly and conversational. "
                "Do NOT write long essays unless the user explicitly asks for a long explanation. "
                "Never say you are a large language model or advanced AI. "
                "Just behave like a normal helpful chat assistant who answers clearly."
            )
        }
    ]

    for sender, msg in history:
        role = "user" if sender == "user" else "assistant"
        messages.append({"role": role, "content": msg})

    messages.append({"role": "user", "content": query})

    data = {
        "model": "deepseek/deepseek-chat",
        "messages": messages,
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]

    return f"API Error: {response.text}"

st.set_page_config(page_title="Chatbot", layout="centered")
st.title("Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

if "just_added" not in st.session_state:
    st.session_state.just_added = None

user_input = st.chat_input("Say something...")

if user_input:
    st.session_state.history.append(("user", user_input))
    intent, confidence = predict_intent(user_input)

    if intent == "ai_handoff":
        bot_reply = deepseek_generate(user_input, st.session_state.history)
    else:
        replies = intent_responses.get(intent, ["I didn't understand that."])
        bot_reply = random.choice(replies)

    st.session_state.history.append(("bot", bot_reply))
    st.session_state.just_added = ("bot", bot_reply)

for sender, msg in st.session_state.history:
    if sender == "user":
        st.chat_message("user").write(msg)

    else:
        if st.session_state.just_added and msg == st.session_state.just_added[1]:
            placeholder = st.chat_message("assistant").empty()
            full = ""
            for word in msg.split():
                full += word + " "
                placeholder.write(full)
                time.sleep(0.04)
            st.session_state.just_added = None 

        else:
            st.chat_message("assistant").write(msg)
