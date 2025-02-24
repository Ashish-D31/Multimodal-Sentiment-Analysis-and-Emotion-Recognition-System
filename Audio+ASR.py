import streamlit as st
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoFeatureExtractor, AutoModelForAudioClassification, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import librosa
import numpy as np
import torch
import io

# ------------------- Load Models Once (Optimized) -------------------

@st.cache_resource
def load_emotion_model():
    model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
    model = AutoModelForAudioClassification.from_pretrained(model_id).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
    id2label = model.config.id2label
    return model, feature_extractor, id2label

@st.cache_resource
def load_transcription_model():
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo").to(device)
    return processor, model

@st.cache_resource
def load_text_pipelines():
    sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")
    emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    return sentiment_pipeline, emotion_pipeline

# ------------------- Global Configurations -------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentiment_pipeline, emotion_pipeline = load_text_pipelines()
speech_emotion_model, feature_extractor, id2label = load_emotion_model()
transcription_processor, transcription_model = load_transcription_model()
MAX_DURATION = 30.0
SAMPLING_RATE = feature_extractor.sampling_rate
MAX_LENGTH = int(SAMPLING_RATE * MAX_DURATION)

# ------------------- Audio Processing -------------------

def load_audio(audio_file):
    """ Load audio from Streamlit uploader without writing to disk. """
    audio_bytes = audio_file.read()
    audio_buffer = io.BytesIO(audio_bytes)
    audio_array, original_sample_rate = librosa.load(audio_buffer, sr=SAMPLING_RATE)
    
    if len(audio_array) > MAX_LENGTH:
        audio_array = audio_array[:MAX_LENGTH]
    else:
        audio_array = np.pad(audio_array, (0, MAX_LENGTH - len(audio_array)))

    return audio_array

def transcribe_audio(audio_array):
    """ Perform transcription on an audio array. """
    inputs = transcription_processor(audio_array, sampling_rate=SAMPLING_RATE, return_tensors="pt")
    with torch.no_grad():
        generated_ids = transcription_model.generate(inputs.input_features.to(device))
    return transcription_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def predict_emotion(audio_array):
    """ Predict emotion from audio array. """
    inputs = feature_extractor(audio_array, sampling_rate=SAMPLING_RATE, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = speech_emotion_model(**inputs).logits
    return id2label[torch.argmax(logits, dim=-1).item()]

def analyze_text(text):
    """ Perform sentiment & emotion analysis in a single function call. """
    sentiment_result, emotion_result = sentiment_pipeline(text), emotion_pipeline(text)
    sentiment_label = ["Negative", "Neutral", "Positive"][int(sentiment_result[0]["label"].split("_")[-1])]
    return sentiment_label, emotion_result[0]["label"]

# ------------------- Streamlit UI -------------------

st.title("🎙️ Speech Analysis: Emotion & Transcription")

audio_file = st.file_uploader("📂 Upload an Audio File", type=["wav", "mp3", "ogg"])

if audio_file:
    with st.spinner("🔄 Processing audio..."):
        audio_array = load_audio(audio_file)

    with st.spinner("📝 Transcribing..."):
        transcription = transcribe_audio(audio_array)
    
    with st.spinner("📊 Analyzing Text..."):
        text_sentiment, text_emotion = analyze_text(transcription)
    
    with st.spinner("🔍 Predicting Speech Emotion..."):
        predicted_emotion = predict_emotion(audio_array)
    
    # ------------------- Display Results -------------------
    
    st.subheader("📝 Transcription")
    st.write(transcription)
    
    st.subheader("📊 Text Analysis")
    st.write(f"**Sentiment:** {text_sentiment}")
    st.write(f"**Emotion:** {text_emotion}")

    st.subheader("🔊 Speech Emotion")
    st.write(f"**Predicted Emotion:** {predicted_emotion}")
