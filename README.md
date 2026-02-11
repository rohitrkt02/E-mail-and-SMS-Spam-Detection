# E-mail-and-SMS-Spam-Detection

![App Screenshot](screenshots/app_screenshots.png)

This is a **web-based Email/SMS Spam Classifier** built using **Streamlit** and **scikit-learn**.  
It allows users to input an email or SMS message and predicts whether it is **Spam** or **Not Spam** using a trained **Multinomial Naive Bayes** model.

## Features

- Classifies messages as **Spam** or **Not Spam**.
- Preprocesses text using:
  - Lowercasing
  - Tokenization
  - Stopwords removal
  - Stemming
- Uses **TF-IDF vectorization** for text features.
- Built with **Streamlit** for a simple and interactive UI.

---

## Requirements

- Python 3.x
- Libraries:
  - `streamlit`
  - `scikit-learn`
  - `nltk`
  - `pickle`

Install the dependencies using:

```
pip install -r requirements.txt
```

## Setup Instructions
1- Clone this repository:
```
git clone <repository_url>
cd <repository_folder>
```

2- Download or create your trained model and vectorizer:

model.pkl → Trained MultinomialNB model

vectorizer.pkl → TF-IDF vectorizer

3- Make sure NLTK resources are downloaded:
```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

4- Run the Streamlit app:
```
streamlit run app.py
```

5- Open the URL provided by Streamlit (usually http://localhost:8501) and start testing messages.

## How It Works
1- User Input: Enter an email or SMS message in the text area.
2- Preprocessing: The message is converted to lowercase, tokenized, stopwords removed, and stemmed.
3- Vectorization: The preprocessed text is transformed using TF-IDF vectorizer.
4- Prediction: The trained Multinomial Naive Bayes model predicts Spam (1) or Not Spam (0).
5- Output: Displays the result in the Streamlit app.

### Sample Messages

- Spam:

"Congratulations! You’ve won a free iPhone. Click here!"

"Win a free trip to Paris! Enter your details now!"

- Not Spam:

"Hey, are we meeting for lunch today?"

"Don't forget to submit your assignment tonight."

## Author

Rohit Kumar Gupta
- Email: rohitkumargupta502@gmail.com