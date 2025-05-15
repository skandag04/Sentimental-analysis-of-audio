from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

# Load the Wav2Vec2 model fine-tuned for emotion recognition
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# Update emotion labels
model.config.id2label = {0: "Neutral", 1: "Happy", 2: "Sad", 3: "Angry"}

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Limit file size to 16MB

def recognize_emotion(audio_path):
    # Load audio file
    signal, fs = torchaudio.load(audio_path)

    # Ensure sampling rate matches the model's requirements
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
        signal = resampler(signal)

    # Normalize audio to mono (if not already)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0)

    # Prepare input for the Wav2Vec2 processor
    inputs = processor(signal.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

    # Perform inference
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Get emotion label and confidence
    emotion = model.config.id2label[predicted_class]
    confidence = torch.softmax(logits, dim=1).max().item()

    return emotion, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            try:
                emotion, confidence = recognize_emotion(file_path)
                return jsonify({
                    "emotion": emotion,
                   # "confidence": f"{confidence * 100:.2f}%"
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5002)
