from flask import Flask, request, jsonify
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

app = Flask(__name__)

@app.route('/resumir', methods=['POST'])
def resumir():
    data = request.get_json()
    texto = data.get('texto', '')
    porcentaje = int(data.get('porcentaje', 30))

    frases = sent_tokenize(texto)
    total = len(frases)
    cantidad = max(1, int(total * (porcentaje / 100)))

    vectorizador = TfidfVectorizer()
    matriz = vectorizador.fit_transform(frases)
    pesos = matriz.sum(axis=1).A1

    conocimiento = [{"frase": f, "peso": p} for f, p in zip(frases, pesos)]
    frases_ordenadas = sorted(conocimiento, key=lambda x: x["peso"], reverse=True)
    seleccionadas = random.sample(frases_ordenadas[:total], cantidad)
    resumen = [f["frase"] for f in seleccionadas]

    return jsonify({"resumen": resumen})

if __name__ == '__main__':
    app.run(port=5000)
