import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, encoding='latin-1')
    df = df.rename(columns={"Category": "label", "Message": "text"})
    df = df[['label', 'text']]
    df = df.dropna(subset=['label', 'text'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df = df.dropna(subset=['label'])

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test