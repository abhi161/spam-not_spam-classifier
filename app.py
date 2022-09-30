import pandas as pd
import numpy as np
import pickle 
import streamlit as st
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import string
import requests
from streamlit_lottie import st_lottie
import warnings
warnings.filterwarnings("ignore")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- LOAD ASSETS ---
lottie_coding = load_lottieurl("https://assets8.lottiefiles.com/private_files/lf30_prqvme9e.json")


tfidf = pickle.load(open('new_vec.pkl','rb'))
model = pickle.load(open('RandomForest.pkl','rb'))
#preprocess = pickle.load(open('preprocessing_text.pkl','rb'))

ps =PorterStemmer()

def processing(text):
        text = text.lower()
        text = nltk.word_tokenize(text)# list of words 
        
        y =[]
        for i in text:
            if i.isalnum():# alpha numeric (removing special characters)
                y.append(i)
                
        #text = y We never copy list this way since list is teh mutable datatype hence we need to do cloning
        text = y[:]
        y.clear()
        
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)
                
        text  = y[:]
        y.clear()
        
        for i in text:
            y.append(ps.stem(i))
            
        return " ".join(y)  # will convert in string

left_column,right_column = st.columns(2)
with left_column:
    st.title('Spam classifier')
with right_column:
    st_lottie(lottie_coding, height=140 )


input = st.text_input('Enter the message')

if st.button('Predict'):

    #1) preprocess text
    transformed = processing(input)
    #2) Vectorize
    vector = tfidf.transform([transformed])
    #3) Predict
    result = model.predict(vector)[0]
    #4) Display
    if result == 0:
        st.write('NOT SPAM')
    else:
        st.write('SPAM')