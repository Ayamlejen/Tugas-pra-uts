
import librosa
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

SAMPLE_RATE = 22050
DURATION = 30
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512


def audio_to_melspectrogram(audio_input):

    # Membaca audio
    y, sr = librosa.load(
        audio_input,
        sr=SAMPLE_RATE,
        mono=True,
        duration=DURATION
    )

    # Menyesuaikan durasi menjadi 30 detik
    target_length = SAMPLE_RATE * DURATION

    if len(y) < target_length:
        y = np.pad(
            y,
            (0, target_length - len(y))
        )
    else:
        y = y[:target_length]

    # Mengubah audio menjadi Mel-Spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Membuat gambar spectrogram
    fig, ax = plt.subplots(figsize=(4, 4))

    ax.imshow(
        mel_db,
        aspect="auto",
        origin="lower",
        cmap="viridis"
    )

    ax.axis("off")
    fig.tight_layout(pad=0)

    fig.canvas.draw()

    # Mengubah visualisasi menjadi gambar RGB
    image = np.asarray(
        fig.canvas.buffer_rgba()
    )[:, :, :3].copy()

    plt.close(fig)

    return Image.fromarray(image)
