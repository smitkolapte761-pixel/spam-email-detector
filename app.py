from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load('models/model.pkl')
tfidf = joblib.load('models/vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_text = data.get('text', '')
    
    # 🚀 NO CLEANING NEEDED! 
    # The vectorizer already knows how to handle lowercasing, punctuation, and stopwords.
    vectorized = tfidf.transform([email_text]) 
    
    # Predict
    pred = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0] 
    
    result = "Spam" if pred == 1 else "Ham"
    confidence = round(proba[1] * 100, 2) if pred == 1 else round(proba[0] * 100, 2)
    
    return jsonify({
        'prediction': result,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)