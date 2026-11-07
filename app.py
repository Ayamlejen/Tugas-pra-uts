
import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np

from PIL import Image
from torchvision import models, transforms
from src.audio_utils import audio_to_melspectrogram

st.set_page_config(
    page_title="Music Genre Classification",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Music Genre Classification")
st.write(
    "Sistem klasifikasi genre musik menggunakan "
    "CNN ResNet18 berdasarkan Audio Mel-Spectrogram."
)

# Konfigurasi perangkat
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Transformasi gambar harus sama dengan saat training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Memuat model
@st.cache_resource
def load_model():
    checkpoint = torch.load(
        "models/resnet18_music.pth",
        map_location=device,
        weights_only=True
    )

    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names

model, class_names = load_model()

# Upload audio
uploaded_file = st.file_uploader(
    "Unggah file musik",
    type=["wav", "mp3"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button("Klasifikasikan Genre Musik"):

        with st.spinner("Sedang menganalisis audio..."):

            try:
                # Mengubah audio menjadi Mel-Spectrogram
                image = audio_to_melspectrogram(uploaded_file)

                # Menampilkan spectrogram
                st.subheader("Visualisasi Mel-Spectrogram")
                st.image(image, use_container_width=True)

                # Persiapan input model
                input_tensor = transform(image)
                input_tensor = input_tensor.unsqueeze(0).to(device)

                # Prediksi
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)

                probabilities = probabilities.cpu().numpy()[0]

                # Genre dengan probabilitas tertinggi
                best_index = np.argmax(probabilities)
                best_genre = class_names[best_index]
                confidence = probabilities[best_index] * 100

                st.success(
                    f"Genre Musik: {best_genre.upper()}"
                )

                st.metric(
                    "Skor probabilitas model",
                    f"{confidence:.2f}%"
                )

                # Top 3 prediksi
                st.subheader("Top 3 Prediksi Genre")

                top_indices = np.argsort(probabilities)[::-1][:3]

                for index in top_indices:
                    st.write(
                        f"{class_names[index].upper()}: "
                        f"{probabilities[index] * 100:.2f}%"
                    )

                    st.progress(float(probabilities[index]))

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
