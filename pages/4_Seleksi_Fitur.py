import streamlit as st
# import library
import pandas as pd
import nltk
import string
import re


st.title('Seleksi Fitur dengan Backward Elimination')

st.header('Data TF-IDF')
df_tfidf= pd.read_csv('hasilpembobotantfidf1200_2.csv', encoding= 'unicode_escape')
st.dataframe(df_tfidf)

st.header('Setelah Proses Backward Elimination')
df_backward= pd.read_csv('hasilbackward1200_belumfix_2.csv', encoding= 'unicode_escape')
st.dataframe(df_backward)