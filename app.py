from flask import Flask, request, jsonify, render_template
import joblib
import re
import os
import nltk
import gdown
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from flask_cors import CORS

# Asigură-te că resursele necesare sunt descărcate
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

# Setăm calea pentru modelul de propagandă
model_dir = "distilbert_propaganda_model"
model_path = os.path.join(model_dir, "model.safetensors")

# Link-ul către modelul de pe Google Drive (folosește link-ul tău specific)
model_url = 'https://drive.google.com/uc?id=1-7Wtfdj1qQM1qCVcbDdxsG1UT12y5o4s'  # Link-ul tău către model

# Funcție pentru descărcarea modelului de pe Google Drive
def download_model():
    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        gdown.download(model_url, model_path, quiet=False)
        print("Modelul a fost descărcat cu succes.")

# Descărcăm modelul dacă nu există local
download_model()

# Încarcă modelul de propagandă și tokenizatorul
propaganda_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
propaganda_tokenizer = AutoTokenizer.from_pretrained(model_dir)
propaganda_model.eval()

# Funcție pentru preprocesare text
def preprocesare_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text

# Funcție de predicție pentru propagandă
def predict_propaganda(text):
    inputs = propaganda_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = propaganda_model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return "Propaganda" if predicted_class == 1 else "Non-Propaganda"

# Ruta pentru afișarea paginii HTML principale
@app.route('/')
def home():
    return render_template('index.html')

# Ruta pentru verificarea articolelor
@app.route('/verifica', methods=['POST'])
def verifica_articol():
    try:
        data = request.get_json()
        titlu = data.get('titlu', '')
        text = data.get('text', '')

        if not titlu or not text:
            return jsonify({"error": "Titlu sau text lipsă"}), 400

        # Preprocesăm titlul și textul
        titlu_text = titlu + ' ' + text
        titlu_text = preprocesare_text(titlu_text)

        # Detectăm dacă textul este propagandă
        rezultat_propaganda = predict_propaganda(titlu_text)

        return jsonify({
            "predictie_propaganda": rezultat_propaganda
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
