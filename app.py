from flask import Flask, request, jsonify, render_template
import joblib
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import nltk
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Asigură-te că resursele necesare sunt descărcate
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

# Încărcăm vectorizatorul și modelul de detectare fake news
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = joblib.load(f)

with open('ensemble_news_classifier_ro.pkl', 'rb') as f:
    model = joblib.load(f)

lemmatizer = WordNetLemmatizer()

# Încarcă modelul personalizat de propagandă și tokenizatorul
propaganda_model_path = r'.\distilbert_propaganda_model'  # Schimbă calea dacă este diferită
propaganda_model = AutoModelForSequenceClassification.from_pretrained(propaganda_model_path)
propaganda_tokenizer = AutoTokenizer.from_pretrained(propaganda_model_path)
propaganda_model.eval()

# Funcție pentru preprocesare text
def preprocesare_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Funcție de predicție propagandă
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
        titlu = data.get('titlu')
        text = data.get('text')

        if not titlu or not text:
            return jsonify({"error": "Titlu sau text lipsă"}), 400

        # Preprocesăm titlul și textul
        titlu_text = titlu + ' ' + text
        titlu_text = preprocesare_text(titlu_text)

        # Verificăm știrea cu modelul Ensemble (fake news)
        text_vec = vectorizer.transform([titlu_text])
        predictie = model.predict(text_vec)
        probabilitati = model.predict_proba(text_vec)

        # Procentajul pentru clasa 'Falsă' și 'Adevărată'
        probabilitate_falsa = probabilitati[0][0] * 100
        probabilitate_adevarata = probabilitati[0][1] * 100
        rezultat_fake_news = 'Adevărată' if predictie[0] == 1 else 'Falsă'

        # Detectăm dacă textul este propagandă folosind modelul personalizat
        rezultat_propaganda = predict_propaganda(text)

        return jsonify({
            "predictie_fake_news": rezultat_fake_news,
            "probabilitate_adevarata": round(probabilitate_adevarata, 2),
            "probabilitate_falsa": round(probabilitate_falsa, 2),
            "predictie_propaganda": rezultat_propaganda
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
