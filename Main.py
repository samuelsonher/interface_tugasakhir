import streamlit as st
# import library
import pandas as pd
import nltk
nltk.download("wordnet")
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

# step 1: case folding

def clean_lower(lwr):
    lwr = lwr.lower() # lowercase text
    return lwr

# step 2: remove punctuation

def remove_punctuation(text):
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return text.translate(translator)

#step 3 Remove numbers
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result#

#step 4 tokenizing
nltk.download('punkt')
def tokenization(text):
    tokens = nltk.tokenize.word_tokenize(text)
    return tokens

#step 5 koreksi kata
normalizad_word = pd.read_excel("normalisasi.xlsx")
normalizad_word_dict = {}

for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[1]

def normalized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]

#step 6 remove stopwords function
stop_factory = StopWordRemoverFactory()
stopwords_sastrawi = list(stop_factory.get_stop_words())
stop_words = stopwords_sastrawi
stop_words = ['yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'dia', 'dua', 'ia', 'seperti', 'jika',
 'jika', 'sehingga', 'kembali', 'dan', 'ini', 'karena', 'kepada', 'oleh', 'saat', 'harus', 'sementara', 'setelah', 'belum',
 'kami', 'sekitar', 'bagi', 'serta', 'di', 'dari', 'telah', 'sebagai', 'masih', 'hal', 'ketika', 'adalah', 'itu', 'dalam',
 'bisa', 'bahwa', 'atau', 'hanya', 'kita', 'dengan', 'akan', 'juga', 'ada', 'mereka', 'sudah', 'saya', 'terhadap',
 'secara', 'agar', 'lain', 'anda', 'begitu', 'mengapa', 'kenapa', 'yaitu', 'yakni', 'daripada', 'itulah', 'lagi',
 'maka', 'tentang', 'demi', 'dimana', 'kemana', 'pula', 'sambil', 'sebelum', 'sesudah', 'supaya', 'guna', 'kah',
 'pun', 'sampai', 'sedangkan', 'selagi', 'sementara', 'tetapi', 'apakah', 'kecuali', 'sebab', 'selain', 'seolah',
 'seraya', 'seterusnya', 'tanpa', 'agak', 'boleh', 'dapat', 'dsb', 'dst', 'dll', 'dahulu', 'dulunya', 'anu',
 'demikian', 'tapi', 'ingin', 'juga', 'nggak', 'mari', 'nanti', 'melainkan', 'oh', 'ok', 'seharusnya', 'sebetulnya',
 'setiap', 'setidaknya', 'sesuatu', 'pasti', 'saja', 'toh', 'ya', 'walau', 'tolong', 'tentu', 'amat', 'apalagi', 'bagaimanapun']

def remove_stopwords(text):
    filtered_text = [word for word in text if word not in stop_words]
    return filtered_text

# step 6 stemming
#remove punct
def remove_punct(text):
    text  = " ".join([char for char in text if char not in string.punctuation])
    return text

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(text):
  text_clean =  stemmer.stem(text)
  return text_clean

# pembobotan TF-IDF

df_start_tfidf = pd.read_csv("hasiltextpre1200_1.csv", encoding='unicode_escape')

X = df_start_tfidf[['content','textPreprocessing']]
y = df_start_tfidf['sentiment']

# calc TF vector
cvect = CountVectorizer()
TF_vector = cvect.fit_transform(X["textPreprocessing"].values.astype('U'))

# normalize TF vector
normalized_TF_vector  = normalize(TF_vector, norm='l1', axis=1)

tfidf    = TfidfVectorizer(norm=None, use_idf=True, smooth_idf=False, sublinear_tf=False)
tfs = tfidf.fit_transform(X['textPreprocessing'].values.astype('U'))

IDF_vector  = tfidf.idf_

tfidf_mat = normalized_TF_vector.multiply(IDF_vector).toarray()

tfidf_vect  = pd.concat([pd.DataFrame(tfidf_mat, columns = tfidf.get_feature_names_out())], axis=1)

df_tfidf = pd.DataFrame(tfidf_mat, columns = tfidf.get_feature_names_out())
df_tfidf['content'] = df_start_tfidf['content']
df_tfidf["textPreprocessing"] = df_start_tfidf["textPreprocessing"]
df_tfidf["sentiment"] = df_start_tfidf["sentiment"]

# pengujian SVM

df_start_svm_be = pd.read_csv('hasilbackward1200_belumfix_1.csv', encoding= 'unicode_escape')
X_train_be, X_test_be, y_train_be, y_test_be = train_test_split(df_start_svm_be.iloc[:,:-3],df_start_svm_be['sentiment'], test_size=0.2)
model_be = SVC(kernel='rbf', C=1, gamma=0.5)
model_be.fit(X_train_be,y_train_be)

df_start_svm = pd.read_csv('hasiltfidf1200_1.csv', encoding= 'unicode_escape')
X_train, X_test, y_train, y_test = train_test_split(df_start_svm.iloc[:,:-3],df_start_svm['sentiment'], test_size=0.2)
model = SVC(kernel='rbf', C=1, gamma=0.5)
model.fit(X_train,y_train)

# halaman klasifikasi

st.title("Analisis Sentimen Ulasan Telkomsel")
with st.form("Analisis Sentimen", clear_on_submit=True):
    data = st.text_area("Masukkan ulasan:  ")
    result = st.form_submit_button("Proses")
    if result:
        #text preprocessing
        data_lower        = clean_lower(data)
        data_clean_punct  = remove_punctuation(data_lower)
        data_rem_number   = remove_numbers(data_clean_punct)
        data_token        = tokenization(data_rem_number)
        data_normalized   = normalized_term(data_token)
        data_stopword     = remove_stopwords(data_normalized)
        data_review       = remove_punct(data_stopword)
        data_stemming     = stemming(data_review)
        data_tp           = [data_stemming]

        #tf-idf
        # calc TF vector
        TF_vector = cvect.transform(data_tp)

        # normalize TF vector
        normalized_TF_vector = normalize(TF_vector, norm='l1', axis=1)

        # calc IDF
        IDF_vector = tfidf.idf_

        # hitung TF x IDF sehingga dihasilkan TFIDF matrix / vector
        tfidf_mat = normalized_TF_vector.multiply(IDF_vector)
        tf_vect   = pd.concat([pd.DataFrame(tfidf_mat.toarray(), columns = tfidf.get_feature_names_out())], axis=1)

        # klasifikasi tanpa backward elimination
        kolom_baru = pd.DataFrame(columns = X_train.columns)
        data_baru = pd.DataFrame()
        for i in tf_vect:
            for j in kolom_baru:
                if i == j:
                    data_baru = pd.concat((data_baru, tf_vect[i]), axis=1)
        my_pred = model.predict(data_baru)

        # klasifikasi dengan backward elimination
        kolom_baru = pd.DataFrame(columns = X_train_be.columns)
        data_baru = pd.DataFrame()
        for i in tf_vect:
            for j in kolom_baru:
                if i == j:
                    data_baru = pd.concat((data_baru, tf_vect[i]), axis=1)
        my_pred_be = model_be.predict(data_baru)

        st.markdown("Ulasan: ")
        st.write(data)
        st.write("Hasil analisis sentimen dengan SVM dan Backward Elimination")
        st.write(f"Sentimen: {my_pred_be}")

        st.write("Hasil analisis sentimen dengan SVM")
        st.write(f"Sentimen: {my_pred}")
    
