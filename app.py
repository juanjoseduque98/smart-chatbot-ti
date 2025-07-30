from flask import Flask, render_template, request
import json
import random
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import os

# Forzar ruta personalizada de NLTK si no encuentra punkt
nltk_data_path = os.path.expanduser('C:/Users/JUANJO/AppData/Roaming/nltk_data')
nltk.data.path.append(nltk_data_path)

# Inicializamos Flask
app = Flask(__name__)

# Cargamos el modelo entrenado
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Cargamos el vectorizador (palabras y etiquetas)
with open('vectorizer.pkl', 'rb') as f:
    data = pickle.load(f)
    all_words = data['all_words']
    tags = data['tags']

# Cargamos las intenciones
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Lematizador de NLTK
lemmatizer = WordNetLemmatizer()

# Funcion para convertir frase a vector (bag of words)
def bag_of_words(sentence, all_words):
    tokens = word_tokenize(sentence.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalnum()]
    return np.array([1 if w in tokens else 0 for w in all_words])

# Funcion para predecir la intencion
def predict_intent(sentence):
    bow = bag_of_words(sentence, all_words)
    prediction = model.predict([bow])[0]
    return tags[prediction]

# Funcion para obtener una respuesta
def get_response(intent):
    for intent_data in intents['intents']:
        if intent_data['tag'] == intent:
            return random.choice(intent_data['responses'])

# Ruta principal: muestra la pagina web
@app.route('/')
def home():
    return render_template('index.html')

# Ruta POST: recibe el mensaje del usuario y responde
@app.route('/get', methods=['POST'])
def chatbot_reply():
    user_input = request.form['msg']
    intent = predict_intent(user_input)
    response = get_response(intent)
    return response

# Ejecutamos el servidor en modo debug
if __name__ == '__main__':
    app.run(debug=True)
