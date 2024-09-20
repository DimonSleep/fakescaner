from flask import Flask, request, jsonify, render_template
import joblib
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import os
import nltk

# Asigură-te că resursele necesare sunt descărcate
nltk.download('wordnet')
nltk.download('stopwords')

# Continuarea codului aplicației tale...


app = Flask(__name__)

# Încărcăm vectorizatorul și modelul
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = joblib.load(f)

with open('ensemble_news_classifier_ro.pkl', 'rb') as f:
    model = joblib.load(f)

lemmatizer = WordNetLemmatizer()

# Funcție pentru preprocesare text
def preprocesare_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Ruta pentru afișarea paginii HTML
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

        # Verificăm știrea cu modelul Ensemble
        text_vec = vectorizer.transform([titlu_text])
        predictie = model.predict(text_vec)
        probabilitati = model.predict_proba(text_vec)

        # Procentajul pentru clasa 'Falsă' și 'Adevărată'
        probabilitate_falsa = probabilitati[0][0] * 100
        probabilitate_adevarata = probabilitati[0][1] * 100

        rezultat = 'Adevărată' if predictie[0] == 1 else 'Falsă'

        return jsonify({
            "predictie": rezultat,
            "probabilitate_adevarata": round(probabilitate_adevarata, 2),
            "probabilitate_falsa": round(probabilitate_falsa, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ruta pentru gestionarea feedback-ului
@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        feedback_data = request.get_json()

        # Citim datele existente din feedback.json
        try:
            with open('feedback.json', 'r') as f:
                feedback_list = json.load(f)
        except FileNotFoundError:
            feedback_list = []

        # Adăugăm noul feedback
        feedback_list.append(feedback_data)

        # Salvăm toate datele înapoi în feedback.json
        with open('feedback.json', 'w') as f:
            json.dump(feedback_list, f, indent=4)

        return jsonify({"message": "Feedback trimis cu succes!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

