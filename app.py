from flask import Flask, request, jsonify, render_template
import joblib
import re
import os
import psycopg2
from psycopg2 import sql
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask_cors import CORS
from huggingface_hub import HfApi

# Configurare token Hugging Face pentru autentificare și acces la modelul privat
token = "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"  # Token-ul tău personal
model_name = "dimonsleep/antipropaganda"

# Asigură-te că resursele necesare sunt descărcate
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

# Încărcare model și vectorizator pentru detectarea știrilor false
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = joblib.load(f)

with open('ensemble_news_classifier_ro.pkl', 'rb') as f:
    model_false_news = joblib.load(f)

lemmatizer = WordNetLemmatizer()

# Încărcare model Hugging Face pentru detectarea propagandei
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model_propaganda = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=token)
model_propaganda.eval()

# Funcție pentru preprocesare text
def preprocesare_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Conectare la PostgreSQL
def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv('PGHOST'),
        database=os.getenv('PGDATABASE'),
        user=os.getenv('PGUSER'),
        password=os.getenv('PGPASSWORD')
    )
    return conn

# Creare tabelă pentru feedback (rulăm o dată pentru a crea tabelul)
def create_feedback_table():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            titlu TEXT,
            text TEXT,
            argument TEXT,
            sursa TEXT,
            nume TEXT,
            email TEXT,
            telefon TEXT
        )
    ''')
    conn.commit()
    cur.close()
    conn.close()

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

        # Preprocesăm titlul și textul pentru verificarea știrilor false
        articol = titlu + ' ' + text
        articol_proc = preprocesare_text(articol)
        text_vec = vectorizer.transform([articol_proc])
        predictie_false_news = model_false_news.predict(text_vec)
        probabilitati_false_news = model_false_news.predict_proba(text_vec)

        probabilitate_falsa = probabilitati_false_news[0][0] * 100
        probabilitate_adevarata = probabilitati_false_news[0][1] * 100
        rezultat_false_news = 'Adevărată' if predictie_false_news[0] == 1 else 'Falsă'

        # Verificare propagandă
        inputs = tokenizer(articol, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model_propaganda(**inputs)
            logits = outputs.logits
            predictie_propaganda = torch.argmax(logits, dim=1).item()
        rezultat_propaganda = "Propaganda" if predictie_propaganda == 1 else "Non-Propaganda"

        return jsonify({
            "predictie_false_news": rezultat_false_news,
            "probabilitate_adevarata": round(probabilitate_adevarata, 2),
            "probabilitate_falsa": round(probabilitate_falsa, 2),
            "predictie_propaganda": rezultat_propaganda
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ruta pentru gestionarea feedback-ului
@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        feedback_data = request.get_json()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            '''
            INSERT INTO feedback (titlu, text, argument, sursa, nume, email, telefon)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''',
            (
                feedback_data['titlu'],
                feedback_data['text'],
                feedback_data['argument'],
                feedback_data['sursa'],
                feedback_data['nume'],
                feedback_data['email'],
                feedback_data['telefon']
            )
        )
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"message": "Feedback trimis cu succes!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    create_feedback_table()  # Creăm tabelul la pornirea aplicației
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
