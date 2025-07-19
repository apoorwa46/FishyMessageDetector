from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import classify_message

app = Flask(__name__)
CORS(app)  # Enable CORS so frontend (JavaScript) can call this

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message')

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    label, confidence = classify_message(message)
    return jsonify({
        'prediction': label,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
