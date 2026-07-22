import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load the CSV
df = pd.read_csv('data/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']] 
df.columns = ['label', 'text'] 
df['label'] = df['label'].map({'ham': 0, 'spam': 1}) 

print("First 5 rows of data:")
print(df.head())
print("\nTotal emails:", len(df))



# Cell 2: Vectorize and Train (COMPLETELY NLTK-FREE)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

X = df['text']
y = df['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 💥 THIS ONE LINE REPLACES YOUR ENTIRE clean_text FUNCTION 💥
# It lowers text, removes punctuation, removes stopwords, and splits words ALL at once!
vectorizer = CountVectorizer(lowercase=True, stop_words='english')

# Convert text to numbers
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Check performance
y_pred = model.predict(X_test_vec)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Cell 3: Convert to numbers and Train (100% NLTK FREE)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1. Create the 'models' folder if it doesn't exist (to avoid errors later)
if not os.path.exists('models'):
    os.makedirs('models')

# 2. Convert text to numbers
# We use df['text'] directly! No need for a separate 'clean_text' column.
# TfidfVectorizer handles lowercasing, stopwords, and punctuation for you.
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(df['text']).toarray()
y = df['label'].values

# 3. Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the model
# Added max_iter=1000 to make sure the model converges properly
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Test the model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {acc*100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# 6. Save the model and vectorizer to the 'models' folder
joblib.dump(model, 'models/model.pkl')
joblib.dump(tfidf, 'models/vectorizer.pkl')
print("\n✅ Model and Vectorizer saved successfully to the 'models/' folder!")