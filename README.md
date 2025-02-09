# Amazon Sarcasm Detector 🛒🤖

This project aims to classify sarcasm in Amazon product reviews using Natural Language Processing (NLP) and Machine Learning.  
It leverages **PySpark** for large-scale text processing and **Scikit-learn** for classification.

## 🚀 Features
- Uses pre-existing Kaggle datasets for Amazon reviews
- Cleans and preprocesses text data
- Applies NLP techniques (TF-IDF, sentiment analysis, etc.)
- Uses Machine Learning (Naive Bayes, Logistic Regression, or LLMs)
- Can be deployed as a simple API or web app using **FastAPI / Streamlit**

---

## 🏃‍♂️ Quick Start
```bash
# Clone the repository
git clone https://github.com/Nut-ell/amazon-sarcasm-detector.git
cd amazon-sarcasm-detector

# Install dependencies
pip install -r requirements.txt

# Train the sarcasm detection model
python train_model.py  

# Run API or web app
python app.py   
