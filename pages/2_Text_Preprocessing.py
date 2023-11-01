import streamlit as st
# import library
import pandas as pd
import nltk
import string
import re

st.title('Text Preprocessing')
df_textpre= pd.read_csv('hasiltextpreprocessing1200_2.csv', encoding= 'unicode_escape')
st.header('Data Ulasan')
st.dataframe(df_textpre['Ulasan'])
st.header('Case Folding')
st.dataframe(df_textpre['Case Folding'])
st.header('Remove Punctuation')
st.dataframe(df_textpre['Remove Punctuation'])
st.header('Remove Number')
st.dataframe(df_textpre['Remove Number'])
st.header('Tokenization')
st.dataframe(df_textpre['Tokenization'])
st.header('Spelling Correction')
st.dataframe(df_textpre['Spelling Correction'])
st.header('Stopwords')
st.dataframe(df_textpre['Stopwords'])
st.header('Stemming')
st.dataframe(df_textpre['Stemming'])