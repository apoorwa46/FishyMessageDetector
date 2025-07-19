import pandas as pd
import string
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Stopwords
stop_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a",
    "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "don", "should", "now"
])

def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    return ' '.join([word for word in text.split() if word not in stop_words])

# Load dataset
df = pd.read_csv("sms.tsv", sep='\t', names=["label", "message"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Reinforcement spam training: repeat phishing phrases many times
dangerous_spam = [
    "Your account has been suspended. Verify it immediately to avoid closure: http://fakebank.com/login",
    "Click here for a free Netflix subscription – only available today!",
    "Urgent: Your PayPal account has been compromised. Log in to fix it now.",
    "You've been selected for a cash prize. Reply with your bank info to claim.",
    "Your computer is infected! Download antivirus now.",
    "Get rich quick! Make $5,000/week from home with no experience.",
    "Legal notice: Failure to respond will result in legal action.",
    "This is your final warning. Act now to avoid penalty."
]

reinforced = pd.DataFrame({
    "label": [1] * len(dangerous_spam) * 500,  # repeat each 500x
    "message": dangerous_spam * 500
})

df = pd.concat([df, reinforced], ignore_index=True)
df["cleaned"] = df["message"].apply(preprocess)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df["cleaned"], df["label"], test_size=0.2, random_state=42)

# Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=2000))
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "spam_model.pkl")
print("✅ Reinforced model saved as spam_model.pkl")
