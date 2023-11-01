import streamlit as st
# import library
import pandas as pd
import nltk
import string
import re
import numpy as np
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
import matplotlib.pyplot as plt 

model = SVC(kernel='rbf', C=1, gamma=0.5)
df_start_svm_be = pd.read_csv('hasilbackward1200_belumfix_2.csv', encoding= 'unicode_escape')
X = df_start_svm_be.iloc[:,:-3]
Y = df_start_svm_be.iloc[:,-1]
score_svm = cross_val_score(model, X, Y, cv=5)
pred = cross_val_predict(model, X, Y, cv=5)
conf_mat = confusion_matrix(Y, pred)
clas_report = classification_report(Y, pred, output_dict=True)
accuracy = accuracy_score(Y, pred)
precision = precision_score(Y, pred, average=None)
recall = recall_score(Y, pred, average=None)
f1score = f1_score(Y, pred, average=None)
df_confmat_be = pd.DataFrame(conf_mat)
df_confmat_be.columns = ['Prediksi Negatif','Prediksi Netral','Prediksi Positif']
df_confmat_be.index = ['Aktual Negatif','Aktual Netral','Aktual Positif']
df_accuracy_be = pd.DataFrame(score_svm)
df_accuracy_be.columns = ['Accuracy BE dan SVM']
df_classrep_be = pd.DataFrame(clas_report).transpose()
df_classrep_be = df_classrep_be.loc[['NEGATIF','NETRAL','POSITIF']]
df_classrep_be = df_classrep_be.drop(columns = ['support'])

df_svm_be = pd.DataFrame(df_start_svm_be['content'])
df_svm_be['Text Preprocessing'] = df_start_svm_be['textPreprocessing']
df_svm_be['Sentimen'] = df_start_svm_be['sentiment']
df_svm_be['Prediksi'] = pred

st.header("Pengujian SVM dan Backward Elimination")
st.subheader('Confusion Matrix')
st.table(df_confmat_be)
st.subheader('Classification Report')
st.write(f"Accuracy     : {clas_report['accuracy']}")
st.table(df_classrep_be)
st.subheader('Data Training dan Testing')
st.dataframe(df_svm_be)

model = SVC(kernel='rbf', C=1, gamma=0.5)
df_start_svm = pd.read_csv('hasiltfidf1200_2.csv', encoding= 'unicode_escape')
X = df_start_svm.iloc[:,:-3]
Y = df_start_svm.iloc[:,-1]
score_svm = cross_val_score(model, X, Y, cv=5)
pred = cross_val_predict(model, X, Y, cv=5)
conf_mat = confusion_matrix(Y, pred)
clas_report = classification_report(Y, pred, output_dict=True)
accuracy = accuracy_score(Y, pred)
precision = precision_score(Y, pred, average=None)
recall = recall_score(Y, pred, average=None)
f1score = f1_score(Y, pred, average=None)
df_confmat = pd.DataFrame(conf_mat)
df_confmat.columns = ['Prediksi Negatif','Prediksi Netral','Prediksi Positif']
df_confmat.index = ['Aktual Negatif','Aktual Netral','Aktual Positif']
df_accuracy = pd.DataFrame(score_svm)
df_accuracy.columns = ['Accuracy SVM']
df_classrep = pd.DataFrame(clas_report).transpose()
df_classrep = df_classrep.loc[['NEGATIF','NETRAL','POSITIF']]
df_classrep = df_classrep.drop(columns = ['support'])

df_svm = pd.DataFrame(df_start_svm['content'])
df_svm['Text Preprocessing'] = df_start_svm['textPreprocessing']
df_svm['Sentimen'] = df_start_svm['sentiment']
df_svm['Prediksi'] = pred

st.header("Pengujian SVM")
st.subheader('Confusion Matrix')
st.table(df_confmat)
st.subheader('Classification Report')
st.write(f"Accuracy     : {clas_report['accuracy']}")
st.table(df_classrep)
st.subheader('Data Training dan Testing')
st.dataframe(df_svm)

st.header('Perbandingan Classification Report (Laporan Pengujian)')
# set width of bar 
barWidth = 0.2

st.subheader('Perbandingan Accuracy (Akurasi)')
fig, ax = plt.subplots(figsize =(12, 8)) 
# set height of bar 
svm = df_accuracy['Accuracy SVM']
be_svm = df_accuracy_be['Accuracy BE dan SVM']

# Set position of bar on X axis 
br1 = np.arange(len(svm)) 
br2 = [x + barWidth for x in br1] 
# Make the plot
ax = plt.bar(br1, svm, color ='r', width = barWidth, label ='SVM') 
ax = plt.bar(br2, be_svm, color ='g', width = barWidth, label ='BE SVM') 
# Adding Xticks 
plt.xlabel('Accuracy', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(svm))],  ['k1','k2','k3','k4','k5'])

plt.legend()
st.pyplot(fig)

st.subheader('Perbandingan Precision (Presisi)')
fig, ax = plt.subplots(figsize =(12, 8)) 
# set height of bar 
svm = df_classrep['precision']
be_svm = df_classrep_be['precision']

# Set position of bar on X axis 
br1 = np.arange(len(svm)) 
br2 = [x + barWidth for x in br1] 
# Make the plot
ax = plt.bar(br1, svm, color ='r', width = barWidth, label ='SVM') 
ax = plt.bar(br2, be_svm, color ='g', width = barWidth, label ='BE SVM') 
# Adding Xticks 
plt.xlabel('Precision', fontweight ='bold', fontsize = 10) 
plt.xticks([r + barWidth for r in range(len(svm))],  df_classrep.index)

plt.legend()
st.pyplot(fig)

st.subheader('Perbandingan Recall')
fig, ax = plt.subplots(figsize =(12, 8)) 
# set height of bar 
svm = df_classrep['recall']
be_svm = df_classrep_be['recall']

# Set position of bar on X axis 
br1 = np.arange(len(svm)) 
br2 = [x + barWidth for x in br1] 
# Make the plot
ax = plt.bar(br1, svm, color ='r', width = barWidth, label ='SVM') 
ax = plt.bar(br2, be_svm, color ='g', width = barWidth, label ='BE SVM') 
# Adding Xticks 
plt.xlabel('Recall', fontweight ='bold', fontsize = 10) 
plt.xticks([r + barWidth for r in range(len(svm))],  df_classrep.index)

plt.legend()
st.pyplot(fig)

st.subheader('Perbandingan F1-Score')
fig, ax = plt.subplots(figsize =(12, 8)) 
# set height of bar 
svm = df_classrep['f1-score']
be_svm = df_classrep_be['f1-score']

# Set position of bar on X axis 
br1 = np.arange(len(svm)) 
br2 = [x + barWidth for x in br1] 
# Make the plot
ax = plt.bar(br1, svm, color ='r', width = barWidth, label ='SVM') 
ax = plt.bar(br2, be_svm, color ='g', width = barWidth, label ='BE SVM') 
# Adding Xticks 
plt.xlabel('F1-Score', fontweight ='bold', fontsize = 10) 
plt.xticks([r + barWidth for r in range(len(svm))],  df_classrep.index)

plt.legend()
st.pyplot(fig)