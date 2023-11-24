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
from sklearn.model_selection import StratifiedKFold

model = SVC(kernel='rbf', C=1, gamma=0.5)
df_start_svm_be = pd.read_csv('hasilbackward1200_belumfix_2.csv', encoding= 'unicode_escape')
x = df_start_svm_be.iloc[:,:-3]
y = df_start_svm_be.iloc[:,-1]

akurasi = []
pre_re_f1 = []
df_traintest_besvm = pd.DataFrame()
skf = StratifiedKFold(n_splits=5)
for train, test in skf.split(x, y):
    X_train, X_test = x.iloc[train], x.iloc[test]
    y_train, y_test = y.loc[train], y.loc[test]
    model = SVC(kernel='rbf', C=1, gamma=0.5)
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    df_pred= pd.DataFrame(df_start_svm_be.loc[test,['content','textPreprocessing','sentiment']])
    df_pred['prediksi'] = prediction
    df_traintest_besvm = pd.concat([df_traintest_besvm, df_pred])
    cr = classification_report(y_test, prediction,output_dict=True)
    akurasi.append(cr['accuracy'])
    pre_re_f1.append(cr['macro avg'])
class_report_be = classification_report(df_traintest_besvm['sentiment'], df_traintest_besvm['prediksi'], output_dict=True)
df_classrep_be = pd.DataFrame(class_report_be).transpose()
df_classrep_be = df_classrep_be.loc[['NEGATIF','NETRAL','POSITIF','macro avg']]
df_classrep_be = df_classrep_be.drop(columns = ['support'])
df_classrep_be = df_classrep_be.mul(100)
conf_matrix = confusion_matrix(df_traintest_besvm['sentiment'], df_traintest_besvm['prediksi'])
df_confmat_be = pd.DataFrame(conf_matrix)
df_confmat_be.columns = ['Prediksi Negatif','Prediksi Netral','Prediksi Positif']
df_confmat_be.index = ['Aktual Negatif','Aktual Netral','Aktual Positif']
akurasi = pd.DataFrame(akurasi, columns=['accuracy'])
pre_re_f1 = pd.DataFrame(pre_re_f1)
pre_re_f1 = pre_re_f1.drop(columns = ['support'])
kfoldreport_besvm = pd.concat([akurasi, pre_re_f1], axis = 1)
mean = pd.DataFrame(kfoldreport_besvm.mean(), columns=['rata-rata']).transpose()
kfoldreport_besvm = pd.concat([kfoldreport_besvm, mean])
kfoldreport_besvm = kfoldreport_besvm.mul(100)

st.header("Pengujian SVM dan Backward Elimination")
st.subheader('Confusion Matrix')
st.table(df_confmat_be)
st.subheader('Classification Report')
st.write(f"Accuracy     : {class_report_be['accuracy']*100}")
st.table(df_classrep_be)
st.subheader('K-Fold Cross Validation')
st.table(kfoldreport_besvm)
st.subheader('Data Training dan Testing')
st.dataframe(df_traintest_besvm)

model = SVC(kernel='rbf', C=1, gamma=0.5)
df_start_svm = pd.read_csv('hasiltfidf1200_2.csv', encoding= 'unicode_escape')
x = df_start_svm.iloc[:,:-3]
y = df_start_svm.iloc[:,-1]

akurasi = []
pre_re_f1 = []
df_traintest_svm = pd.DataFrame()
skf = StratifiedKFold(n_splits=5)
for train, test in skf.split(x, y):
    X_train, X_test = x.iloc[train], x.iloc[test]
    y_train, y_test = y.loc[train], y.loc[test]
    model = SVC(kernel='rbf', C=1, gamma=0.5)
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    df_pred= pd.DataFrame(df_start_svm.loc[test,['content','textPreprocessing','sentiment']])
    df_pred['prediksi'] = prediction
    df_traintest_svm = pd.concat([df_traintest_svm, df_pred])
    cr = classification_report(y_test, prediction,output_dict=True)
    akurasi.append(cr['accuracy'])
    pre_re_f1.append(cr['macro avg'])
class_report = classification_report(df_traintest_svm['sentiment'], df_traintest_svm['prediksi'], output_dict=True)
df_classrep = pd.DataFrame(class_report).transpose()
df_classrep = df_classrep.loc[['NEGATIF','NETRAL','POSITIF','macro avg']]
df_classrep = df_classrep.drop(columns = ['support'])
df_classrep = df_classrep.mul(100)
conf_matrix = confusion_matrix(df_traintest_svm['sentiment'], df_traintest_svm['prediksi'])
df_confmat = pd.DataFrame(conf_matrix)
df_confmat.columns = ['Prediksi Negatif','Prediksi Netral','Prediksi Positif']
df_confmat.index = ['Aktual Negatif','Aktual Netral','Aktual Positif']
akurasi = pd.DataFrame(akurasi, columns=['accuracy'])
pre_re_f1 = pd.DataFrame(pre_re_f1)
pre_re_f1 = pre_re_f1.drop(columns = ['support'])
kfoldreport_svm = pd.concat([akurasi, pre_re_f1], axis = 1)
mean = pd.DataFrame(kfoldreport_svm.mean(), columns=['rata-rata']).transpose()
kfoldreport_svm = pd.concat([kfoldreport_svm, mean])
kfoldreport_svm = kfoldreport_svm.mul(100)

st.header("Pengujian SVM")
st.subheader('Confusion Matrix')
st.table(df_confmat)
st.subheader('Classification Report')
st.write(f"Accuracy     : {class_report['accuracy']*100}")
st.table(df_classrep)
st.subheader('K-Fold Cross Validation')
st.table(kfoldreport_svm)
st.subheader('Data Training dan Testing')
st.dataframe(df_traintest_svm)

st.header('Perbandingan Classification Report (Laporan Pengujian)')
# set width of bar 
barWidth = 0.2

st.subheader('Perbandingan Accuracy (Akurasi)')
fig, ax = plt.subplots(figsize =(12, 8)) 
# set height of bar 
svm = kfoldreport_svm['accuracy']
be_svm = kfoldreport_besvm['accuracy']

# Set position of bar on X axis 
br1 = np.arange(len(svm)) 
br2 = [x + barWidth for x in br1] 
# Make the plot
bars = ax.bar(br1, svm, color ='r', width = barWidth, label ='SVM') 
bars = ax.bar(br2, be_svm, color ='g', width = barWidth, label ='BE SVM') 
# Adding Xticks 
plt.xlabel('Accuracy', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(svm))],  ['k1','k2','k3','k4','k5','rata-rata'])

for bars in ax.containers:
    ax.bar_label(bars)

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
bars = ax.bar(br1, svm, color ='r', width = barWidth, label ='SVM') 
bars = ax.bar(br2, be_svm, color ='g', width = barWidth, label ='BE SVM') 
# Adding Xticks 
plt.xlabel('Precision', fontweight ='bold', fontsize = 10) 
plt.xticks([r + barWidth for r in range(len(svm))],  df_classrep.index)

for bars in ax.containers:
    ax.bar_label(bars)

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
bars = ax.bar(br1, svm, color ='r', width = barWidth, label ='SVM') 
bars = ax.bar(br2, be_svm, color ='g', width = barWidth, label ='BE SVM') 
# Adding Xticks 
plt.xlabel('Recall', fontweight ='bold', fontsize = 10) 
plt.xticks([r + barWidth for r in range(len(svm))],  df_classrep.index)

for bars in ax.containers:
    ax.bar_label(bars)

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
bars = ax.bar(br1, svm, color ='r', width = barWidth, label ='SVM') 
bars = ax.bar(br2, be_svm, color ='g', width = barWidth, label ='BE SVM') 
# Adding Xticks 
plt.xlabel('F1-Score', fontweight ='bold', fontsize = 10) 
plt.xticks([r + barWidth for r in range(len(svm))],  df_classrep.index)

for bars in ax.containers:
    ax.bar_label(bars)

plt.legend()
st.pyplot(fig)