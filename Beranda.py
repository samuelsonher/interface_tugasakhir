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

st.title("Penerapan Support Vector Machine dengan Seleksi Fitur Backward Elimination untuk Analisis Sentimen Kepuasan Pelanggan Operator Seluler Telkomsel")
