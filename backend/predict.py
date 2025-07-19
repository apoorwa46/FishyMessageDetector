import joblib
import string
import numpy as np
import re

# Minimal stopword list
stop_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "can", "will", "just", "don", "should", "now"
])

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    return ' '.join([w for w in text.split() if w not in stop_words])

# Load model
model_pipeline = joblib.load("spam_model.pkl")

# ⚠️ Hybrid classification: rule + ML
def classify_message(text):
    # Rule-based spam override
    phishing_keywords = [
        "account has been suspended", "verify it immediately", "click here", "login to fix",
        "your account has been compromised", "claim your prize", "update your info", "urgent",
        "action required", "bank info", "tax refund", "free netflix", "password reset",
        "pending delivery", "device is infected", "security alert", "final notice"
    ]

    phishing_patterns = [
        r"http[s]?://",  # any link
        r"bit\.ly", r"tinyurl\.com",  # shorteners
        r"free\s+\w+",  # free gift, free netflix
        r"verify.*account",
        r"account.*suspended",
        r"login.*fix",
        r"bank.*info",
    ]

    lower = text.lower()
    for keyword in phishing_keywords:
        if keyword in lower:
            return "Spam", 99.99

    for pattern in phishing_patterns:
        if re.search(pattern, lower):
            return "Spam", 99.99

    # Otherwise use model
    clean = preprocess(text)
    proba = model_pipeline.predict_proba([clean])[0]
    label = model_pipeline.predict([clean])[0]
    confidence = round(np.max(proba) * 100, 2)
    return ("Spam" if label == 1 else "Not Spam", confidence)
