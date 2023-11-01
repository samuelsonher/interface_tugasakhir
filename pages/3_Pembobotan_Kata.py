import streamlit as st
# import library
import pandas as pd
import nltk
import string
import re

st.title('Pembobotan Kata')

st.header('Data Ulasan')
df_review= pd.read_csv('hasilpembobotantfidf1200_2.csv', encoding= 'unicode_escape')
st.dataframe(df_review[['Ulasan','Text Preprocessing','Sentimen']])

st.header('Term Frequency')
df_tf= pd.read_csv('termfrequency_2.csv', encoding= 'unicode_escape')
st.dataframe(df_tf)

st.header('Normalisasi Term Frequency')
df_tfnorm= pd.read_csv('tfnormalisasi_2.csv', encoding= 'unicode_escape')
st.dataframe(df_tfnorm)

st.header('Inverse Document Frequency')
df_idf= pd.read_csv('inversedf_2.csv', encoding= 'unicode_escape')
st.dataframe(df_idf)

st.header('TF-IDF')
df_tfidf= pd.read_csv('hasilpembobotantfidf1200_2.csv', encoding= 'unicode_escape')
st.dataframe(df_tfidf.iloc[:,:-3])