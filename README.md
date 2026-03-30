# Spam-Email-Classifier


This project is a Machine Learning based web application that classifies emails as **Spam** or **Not Spam** using Natural Language Processing (NLP) techniques.

The system is trained using a modern email dataset in Parquet format and deployed using a simple and professional web interface built with Streamlit.

---

## Project Overview

Spam emails are one of the most common cybersecurity problems today. This project uses machine learning to automatically detect spam emails based on their content.

The application allows the user to paste any email message and instantly get a prediction along with the confidence level.

---

## Technologies Used

- Python
- Machine Learning
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Logistic Regression
- Streamlit (Web Interface)
- Pandas
- Scikit-learn

---

## Project Structure

Spam-Email-Classifier
│
├── app.py
├── train_model.py
├── spam_model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md


---

## How It Works

1. The dataset is loaded using Pandas.
2. The email text is cleaned using text preprocessing.
3. TF-IDF is used to convert text into numerical features.
4. A Logistic Regression model is trained.
5. The trained model is saved using Pickle.
6. The Streamlit web app loads the model and predicts spam emails in real time.

---

## Features

- Detects Spam and Not Spam emails
- Real-time prediction
- Confidence score display
- Clean and professional UI
- Fast prediction using trained ML model

---

## How to Run the Project

Step 1: Install required libraries
pip install -r requirements.txt

Step 2: Run the application
streamlit run app.py

---

## Example Test Email

Spam Example:
Congratulations! You have won a free iPhone. Click here to claim your reward now.


Not Spam Example:

Hello sir, I am sending the project report you asked for. Please check and let me know.


---

## Future Improvements

- Deep Learning based spam detection
- Email attachment analysis
- Multiple language spam detection
- Deployment using cloud platforms

---

## Arshad
 
Spam Email Classifier 
