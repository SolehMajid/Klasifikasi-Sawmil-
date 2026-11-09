import io
import logging
import os

import librosa
import numpy as np
import pandas as pd
from flask import Blueprint, render_template, request


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_REQUEST_OVERHEAD_BYTES = 64 * 1024
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def create_upload_blueprint(
    model,
    feature_extractor,
    feature_columns,
    sample_rate,
    window_samples,
    hop_samples,
):
    upload_blueprint = Blueprint("upload_classifier", __name__)

    @upload_blueprint.route("/upload", methods=["GET", "POST"])
    def upload_audio():
        if request.method == "GET":
            return render_template("upload.html")

        if request.content_length and request.content_length > (
            MAX_UPLOAD_BYTES + MAX_REQUEST_OVERHEAD_BYTES
        ):
            return render_template(
                "upload.html", error="Ukuran file maksimal 25 MB."
            ), 413

        audio_file = request.files.get("audio")
        if audio_file is None or not audio_file.filename:
            return render_template(
                "upload.html", error="Pilih file audio terlebih dahulu."
            ), 400

        extension = os.path.splitext(audio_file.filename)[1].lower()
        if extension not in ALLOWED_EXTENSIONS:
            return render_template(
                "upload.html",
                error="Format file harus WAV, MP3, FLAC, OGG, atau M4A.",
            ), 400

        if model is None:
            return render_template(
                "upload.html",
                error="Model klasifikasi belum tersedia di server.",
            ), 503

        audio_bytes = audio_file.read(MAX_UPLOAD_BYTES + 1)
        if not audio_bytes:
            return render_template(
                "upload.html", error="File audio kosong."
            ), 400
        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            return render_template(
                "upload.html", error="Ukuran file maksimal 25 MB."
            ), 413

        try:
            audio, _ = librosa.load(
                io.BytesIO(audio_bytes), sr=sample_rate, mono=True
            )
        except Exception:
            logging.exception("Gagal membaca audio yang diunggah")
            return render_template(
                "upload.html",
                error="File tidak dapat dibaca sebagai audio. Coba ekspor ulang ke WAV atau MP3.",
            ), 400

        if len(audio) < window_samples:
            minimum_seconds = window_samples / sample_rate
            return render_template(
                "upload.html",
                error=f"Durasi audio minimal {minimum_seconds:.1f} detik.",
            ), 400

        probabilities = []
        for start in range(0, len(audio) - window_samples + 1, hop_samples):
            window = audio[start:start + window_samples]
            features, _, is_silent = feature_extractor(window, sr=sample_rate)
            if is_silent:
                probabilities.append(0.0)
                continue

            model_input = pd.DataFrame([features])[feature_columns]
            if hasattr(model, "predict_proba"):
                sawmill_probability = float(model.predict_proba(model_input)[0][1])
            else:
                prediction = int(model.predict(model_input)[0])
                sawmill_probability = 1.0 if prediction == 1 else 0.0
            probabilities.append(sawmill_probability)

        average_probability = float(np.mean(probabilities))
        sawmill_percent = round(average_probability * 100, 1)
        non_sawmill_percent = round((1 - average_probability) * 100, 1)
        result = {
            "filename": audio_file.filename,
            "duration": round(len(audio) / sample_rate, 1),
            "windows": len(probabilities),
            "label": "Sawmill" if average_probability >= 0.5 else "Non-Sawmill",
            "confidence": max(sawmill_percent, non_sawmill_percent),
            "sawmill_percent": sawmill_percent,
            "non_sawmill_percent": non_sawmill_percent,
        }
        return render_template("upload.html", result=result)

    return upload_blueprint
