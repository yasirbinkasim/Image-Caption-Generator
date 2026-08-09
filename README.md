# Image-Caption-Generator
An AI-powered multimodal web application built with Flask, BLIP HuggingFace Transformer model, and Google Text-to-Speech (gTTS) for real-time English image captioning, Hindi translation, and voice synthesis.

# 📷 AI Image Caption Generator & Audio Synthesizer

A state-of-the-art Deep Learning web application designed for automated scene understanding, natural language caption generation, multilingual translation, and text-to-speech processing.

---

## ✨ Features

- 🖼️ **Image Scene Processing:** Generates descriptive English captions using Salesforce's BLIP fine-tuned transformer model.
- 🌐 **Hindi Language Support:** Translates generated English captions into Hindi in real-time.
- 🔊 **Voice Synthesizer:** Converts text captions into audio speech outputs (`.mp3`).
- 📊 **Analytics Dashboard:** Tracks prediction history logs with real-time BLEU-4 quality accuracy metrics and options to clear history.
- 🎨 **Modern Cyberpunk UI:** Fully responsive interface built using Tailwind CSS and glassmorphism styling.

---

## 🛠️ Tech Stack & Models

- **Backend Framework:** Python (Flask)
- **Deep Learning Model:** BLIP (`Salesforce/blip-image-captioning-base`) via HuggingFace Transformers & PyTorch
- **Translation:** Deep Translator Engine (`GoogleTranslator`)
- **Speech Synthesis:** gTTS (Google Text-to-Speech)
- **Frontend:** HTML5, Tailwind CSS, FontAwesome, JavaScript

---

## 🚀 Quick Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/Image-Caption-Generator.git](https://github.com/YOUR_USERNAME/Image-Caption-Generator.git)
cd Image-Caption-Generator

-----
## Create Virtual Environment & Activate

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

-----
## Install Dependencies

pip install -r requirements.txt

-------
## Run the Flask Server

python app.py

------
## Project Structure

├── static/
│   ├── uploads/          # Temporary uploaded images
│   └── audio/            # Generated speech audio files
├── templates/
│   ├── index.html        # Main image upload and output page
│   └── dashboard.html    # Analytics & prediction history page
├── app.py                # Core Flask server and AI inference pipeline
├── .gitignore            # Files ignored by Git
├── README.md             # Project documentation
└── requirements.txt      # Required Python packages



