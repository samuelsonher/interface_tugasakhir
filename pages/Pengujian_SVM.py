import streamlit as st
# import library
import pandas as pd
import nltk
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

model = SVC(kernel='rbf', C=1, gamma=0.5)
df_start_svm = pd.read_csv('hasiltfidf1200_2.csv', encoding= 'unicode_escape')
X = df_start_svm.iloc[:,:-3]
Y = df_start_svm.iloc[:,-1]
#score_svm = cross_val_score(model, X, Y, cv=5)
pred = cross_val_predict(model, X, Y, cv=5)
#conf_mat = confusion_matrix(Y, pred)
#clas_report = classification_report(Y, pred)
accuracy = accuracy_score(Y, pred)
precision = precision_score(Y, pred, average=None)
recall = recall_score(Y, pred, average=None)
f1score = f1_score(Y, pred, average=None)
df_svm = pd.DataFrame(df_start_svm['content'])
pd.set_option('display.max_colwidth', 40)
df_svm['textPreprocessing'] = df_start_svm['textPreprocessing']
df_svm['sentiment'] = df_start_svm['sentiment']
df_svm['prediksi'] = pred

st.title("Pengujian SVM")
st.write(f"Accuracy     : {accuracy}")
st.write(f"Precision Negatif, Netral, Positif   : {precision}")
st.write(f"Recall Negatif, Netral, Positif      : {recall}")
st.write(f"F1-Score Negatif, Netral, Positif    : {f1score}")
st.dataframe(df_svm)
