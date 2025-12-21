import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Load the pre-trained model with error handling
@st.cache_resource
def load_model():
    try:
        # Try different possible paths for the model file
        possible_paths = [
            'spam_classifier_model.pkl',
            './spam_classifier_model.pkl',
            os.path.join(os.path.dirname(__file__), 'spam_classifier_model.pkl')
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            st.error("Model file 'spam_classifier_model.pkl' not found in any expected location. Please ensure the model file is uploaded.")
            return None
            
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize model
model = load_model()

# #preprocessing function
# - Lower case
# - Tokenization
# - Removing special characters
# - Removing stop words and punctuation
# - Stemming
# Download required NLTK resources (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove special characters (keep alphanumeric)
    tokens = [word for word in tokens if word.isalnum()]
    
    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    
    # Stemming
    tokens = [ps.stem(word) for word in tokens]
    
    return tokens

#vectorize
#Already in pipeline

#predict
def predict_spam(message):
    if model is None:
        st.error("Model not loaded. Cannot make predictions.")
        return None
    
    # Preprocess the message
    processed_message = preprocess_text(message)
    # Join tokens back to string for vectorization
    processed_message_str = ' '.join(processed_message)
    # Predict using the loaded model
    prediction = model.predict([processed_message_str])
    return prediction[0]
#output
def main():
    st.title("Spam Classifier")
    st.write("Enter a message to classify it as Spam or Ham (Not Spam).")
    
    user_input = st.text_area("Message:")
    
    if st.button("Classify"):
        if user_input:
            result = predict_spam(user_input)
            if result is not None:
                if result == 1:
                    st.error("The message is classified as: SPAM")
                    print("Spam")
                else:
                    st.success("The message is classified as: HAM (Not Spam)")
                    print("Ham")
        else:
            st.warning("Please enter a message to classify.")

if __name__ == "__main__":
    main()