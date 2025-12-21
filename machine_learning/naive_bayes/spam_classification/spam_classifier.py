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

# Load the pre-trained model
model = pickle.load(open('spam_classifier_model.pkl', 'rb'))

# #preprocessing function
# - Lower case
# - Tokenization
# - Removing special characters
# - Removing stop words and punctuation
# - Stemming
# Download required NLTK resources (run once)
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