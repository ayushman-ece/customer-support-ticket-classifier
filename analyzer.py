
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


def load_data(file_path="tickets.csv"):
    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[["ticket_text", "category"]].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    return df


def analyze_data(df):
    print("\nDataset Overview")
    print("----------------")
    print("Total tickets:", len(df))
    print("\nCategory distribution:")
    print(df["category"].value_counts())


def train_model(df):
    X = df["ticket_text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )),
        ("classifier", MultinomialNB())
    ])

    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(
        accuracy_score(y_test, predictions) * 100, 2
    ), "%")

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))


def predict_ticket(model, ticket_text):
    prediction = model.predict([ticket_text])[0]
    return prediction
