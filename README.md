# 🎫 Customer Support Ticket Categorization System

## 📌 Project Overview

The Customer Support Ticket Categorization System is a Python-based Machine Learning project that automatically classifies customer complaints into predefined categories using the Multinomial Naive Bayes algorithm.

It analyzes customer complaint text using TF-IDF vectorization and predicts whether a ticket belongs to Billing, Technical Support, Account Issues, or Delivery.

## 🎯 Objectives

- Analyze customer support tickets.
- Automatically categorize customer complaints.
- Apply supervised Machine Learning.
- Convert text into numerical features using TF-IDF.
- Train a text classification model.
- Evaluate model performance.
- Predict categories for new customer complaints.

## ⚙️ Features

- CSV dataset loading and validation.
- Customer ticket data analysis.
- Category-wise ticket distribution.
- Text preprocessing using TF-IDF.
- Unigram and bigram feature extraction.
- Training and testing data separation.
- Multinomial Naive Bayes classification.
- Machine Learning pipeline implementation.
- Model accuracy evaluation.
- Classification report generation.
- Interactive ticket category prediction.

## 🧠 Workflow

1. Load and validate customer ticket data.
2. Analyze ticket categories.
3. Separate input text and target labels.
4. Split data into training and testing sets.
5. Convert text into numerical features using TF-IDF.
6. Train the Multinomial Naive Bayes model.
7. Evaluate model performance.
8. Accept new customer complaints.
9. Predict the appropriate ticket category.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data manipulation |
| Scikit-learn | Machine Learning |
| TF-IDF | Text vectorization |
| Multinomial Naive Bayes | Text classification |

## 📊 Dataset Information

The sample dataset contains 32 synthetic customer support tickets.

| Feature | Description |
|---|---|
| ticket_id | Unique ticket identifier |
| ticket_text | Customer complaint or query |
| category | Target classification label |

### Ticket Categories

- Billing
- Technical Support
- Account Issues
- Delivery

## 🔍 Text Classification

The project uses TF-IDF (Term Frequency–Inverse Document Frequency) to convert customer complaint text into numerical features.

The Multinomial Naive Bayes algorithm learns patterns from these features and predicts the most likely category.

**Model Configuration:**

- TF-IDF lowercase conversion enabled.
- English stop-word removal.
- Unigram and bigram extraction.
- Multinomial Naive Bayes classifier.
- 75% training data and 25% testing data.
- Random state: 42.
- Stratified train-test splitting.

## 📁 Project Structure

| File | Description |
|---|---|
| main.py | Main application and user interaction |
| analyzer.py | Data processing, model training and prediction |
| tickets.csv | Sample labelled customer ticket dataset |
| requirements.txt | Project dependencies |
| .gitignore | Files excluded from Git tracking |

## 🚀 Installation and Execution

**1. Clone the repository**

```bash
git clone https://github.com/ayushman-ece/customer-support-ticket-classifier.git
```

**2. Open the project folder**

```bash
cd customer-support-ticket-classifier
```

**3. Create a virtual environment**

```bash
python -m venv .venv
```

**4. Activate on Windows**

```bash
.venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Run the project**

```bash
python main.py
```

**7. Enter a customer complaint**

Example:

```text
I was charged twice for my subscription.
```

**Expected category:**

```text
Billing
```

Type `exit` to close the application.

## 📚 Key Concepts Learned

- Supervised Machine Learning
- Natural Language Processing (NLP)
- Text Classification
- TF-IDF Vectorization
- Term Frequency and Inverse Document Frequency
- Unigram and Bigram Features
- Multinomial Naive Bayes
- Train-Test Split
- Scikit-learn Pipelines
- Model Training and Prediction
- Accuracy Score
- Classification Report

## ⚠️ Limitations

- Uses only 32 synthetic customer support tickets.
- Limited vocabulary and ticket categories.
- Predictions may be inaccurate for unfamiliar complaints.
- Does not understand complex context, sarcasm, or multiple issues in one ticket.
- Model performance may not generalize to real-world customer support data.
- Not suitable for production customer support automation without further training and validation.

## 🔮 Future Improvements

- Use a larger real-world customer support dataset.
- Add more ticket categories.
- Implement advanced NLP techniques.
- Compare Naive Bayes with Logistic Regression and other classifiers.
- Add prediction confidence scores.
- Build a Streamlit-based customer support dashboard.
- Integrate the classifier with a customer support chatbot.
- Store and manage customer ticket history.

## 👨‍💻 Author

**Ayushman Tiwari**

GitHub: [@ayushman-ece](https://github.com/ayushman-ece)

---

⭐ If you find this project useful, consider giving the repository a star!