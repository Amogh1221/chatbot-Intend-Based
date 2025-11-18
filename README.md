# 🚀 Intent-Based Chatbot with AI Handoff

A lightweight conversational chatbot that uses a **CNN/LSTM intent classifier** for normal queries and automatically switches to **DeepSeek (via OpenRouter)** for complex questions.  
Built with **Streamlit**, supports word-by-word streaming output, and uses a custom dataset in **Hinglish + English + Hindi**.

---

## ✨ Features

*   **🎯 Intent-based classifier:** Uses a TensorFlow (CNN/LSTM) model for instant responses.
*   **🤖 AI Handoff:** Automatically detects complex queries and forwards them to DeepSeek (LLM).
*   **💬 Streaming Replies:** ChatGPT-style word-by-word output animation.
*   **🌐 Multilingual:** Optimized for English, Hindi, and Hinglish.
*   **📚 Custom Dataset:** Curated JSON dataset optimized for specific intent classification.
*   **🔒 Secure:** Uses `.env` for API key storage.
*   **⚡ Fast & Local:** Minimal latency for standard intents.
*   **🧠 Context Aware:** Maintains chat history memory during the session.

---

## 📁 Project Structure

```text
chatbot/
│── app.py                 # Main chatbot UI (Streamlit)
│── train_model.py         # Script to train the LSTM/CNN model
│── best_model.keras       # Trained intent classifier model
│── tokenizer.pkl          # Fitted tokenizer (saved)
│── label_encoder.pkl      # Label encoder for intents (saved)
│── chatbot_dataset.json   # Custom curated dataset
│── requirements.txt       # Project dependencies
│── .env.example           # Example environment file
│── README.md              # Documentation