import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC

st.title("APLIKASI DATA MINING")
st.write("Author: Nicolaus Andiko Nugroho | 190411100136")
st.write("Data Mining Project for Final Exam")
data_desc, import_data, preporcessing, modeling, implementation = st.tabs(["Deskripsi Data","Import Data", "Prepocessing", "Modeling", "Implementation"])

with data_desc:
    st.write("# Deskripsi Data")
    st.write("Data ini merupakan data prediksi apakah orang ini akan membeli sebuah barang atau tidak. Pada kolom pertama terdapat fitur jenis kelamin, male or female. Kolom kedua merupakan kolom umur. Lalu kolom ketiga merupakan fitur perkiraan gaji. Lalu ada kolom yang akan di-drop yaitu User ID karena bukan merupakan fitur dari data mining.")

with import_data:
    st.write("# Import Data")
    st.write("Data menggunakan data Social Network Ads")
    st.write("Data dapat diunduh di: https://www.kaggle.com/datasets/rakeshrau/social-network-ads?resource=download")
    data = pd.read_csv("social_network_ads.csv", error_bad_lines=False)
    st.dataframe(data)

with preporcessing:
    st.write("""# Preprocessing""")
    st.write("Preprocessing adalah proses menyiapkan data dasar atau inti sebelum melakukan proses lainnya. \nPada dasarnya data preprocessing dapat dilakukan dengan membuang data yang tidak sesuai atau mengubah data menjadi bentuk yang lebih mudah untuk diproses oleh sistem. \nProses pembersihan meliputi penghilangan duplikasi data, pengisian atau penghapusan data yang hilang, pembetulan data yang tidak konsisten, dan pembetulan salah ketik. \nSeperti namanya, normalisasi dapat diartikan secara sederhana sebagai proses menormalkan data dari hal-hal yang tidak sesuai.")
    st.write("Dataset Asli:")
    st.dataframe(data)

    st.write("Drop User ID karena bukan merupakan fitur.")
    data = data.drop(columns=["User ID"])
    st.dataframe(data)
    
    st.write("Karena kolom Purchased adalah target, maka kita pisahkan variabel bebas dan variabel terikat")
    X = data.iloc[:,:3].values
    y = data.iloc[:,3].values
    st.dataframe(X)
    st.dataframe(y)

    st.write("Ubah data 'Gender' dari kategorikal menjadi numerik")
    ct = ColumnTransformer([("Gender", OneHotEncoder(), [0])], remainder='passthrough')
    X = ct.fit_transform(X)
    st.dataframe(X)


# with modeling:
#     st.write("""# Modeling""")
#     naive_bayes, bagging_naive_bayes, bagging_decision_tree = st.tabs(["Naive Bayes", "Bagging Naive Bayes", "Bagging Decision Tree"])

#     with naive_bayes:
#         st.write("## Naive Bayes")
#         naive_bayes_classifier = GaussianNB()
#         naive_bayes_classifier.fit(X_train, y_train)
#         Y_pred_nb = naive_bayes_classifier.predict(X_test)


#         ### Making the confusion matrix
#         cm = confusion_matrix(y_test, Y_pred_nb)


#         ### Printing the accuracy, precision, and recall of the model
#         st.write('Confusion matrix for Gaussian Naive Bayes\n',cm)

#         naive_bayes_accuracy = round(100 * accuracy_score(y_test, Y_pred_nb), 2)
#         model_accuracy['Gaussian Naive Bayes'] = naive_bayes_accuracy

#         naive_bayes_precision = round(100 * precision_score(y_test, Y_pred_nb, average = 'weighted'), 2)
#         model_precision['Gaussian Naive Bayes'] = naive_bayes_precision

#         naive_bayes_recall = round(100 * recall_score(y_test, Y_pred_nb, average = 'weighted'), 2)
#         model_recall['Gaussian Naive Bayes'] = naive_bayes_recall

#         st.write('The accuracy of this model is {} %.'.format(naive_bayes_accuracy))
#         st.write('The precision of this model is {} %.'.format(naive_bayes_precision))
#         st.write('The recall of this model is {} %.'.format(naive_bayes_recall))

#         st.write("## Menampilkan prediksi dengan Naive Bayes")
#         st.write(Y_pred_nb)

#     with bagging_naive_bayes:
#         st.write("## Bagging Naive Bayes")
#         clf = BaggingClassifier(base_estimator=GaussianNB(),n_estimators=10, random_state=0).fit(X_train, y_train)
#         rsc = clf.predict(X_test)
#         c = ['Naive Bayes']
#         Bayes = pd.DataFrame(rsc,columns = c)

#         st.write("Menampilkan hasil prediksi dengan esamble naive bayes")
#         st.write(Bayes)

#         bagging_Bayes = round(100 * accuracy_score(y_test, Bayes), 2)
#         st.write('The accuracy of this model is Bagging Naive Bayes {} %.'.format(bagging_Bayes))
        
#     with bagging_decision_tree:
#         st.write("## Bagging Decision Tree")
#         clf_tree = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=10, random_state=0).fit(X_train, y_train)
#         rsc = clf_tree.predict(X_test)
#         c = ['Decision Tree']
#         tree = pd.DataFrame(rsc,columns = c)

#         st.write("Menampilkan hasil prediksi dengan Essamble Decision Tree")
#         st.write(tree)

#         bagging_tree = round(100 * accuracy_score(y_test, tree), 2)
#         st.write('The accuracy of this model is Bagging Decision Tree {} %.'.format(bagging_tree))



#     FIRST_IDX = 0

# with implementation:
#     st.write("# Implementation")
#     nama_nasabah = st.text_input('Masukkan Nama Nasabah')
#     pendapatan_per_tahun = st.number_input('Masukkan pendapatan pertahun')
#     durasi_peminjaman = st.number_input('Masukkan Durasi Peminjaman')
#     jumlah_tanggungan = st.number_input('Masukkan Jumlah Tanggungan')

#     clf = GaussianNB()
#     clf.fit(matrices_X, matrices_Y)
#     clf_pf = GaussianNB()
#     clf_pf.partial_fit(matrices_X, matrices_Y, np.unique(matrices_Y))

#     cek_rasio_NB = st.button('Cek Risk Ratio dengan Naive Bayes')
#     cek_rasio_BNB = st.button('Cek Risk Ratio dengan Bagging Naive Bayes')
#     cek_rasio_DC = st.button('Cek Risk Ratio dengan Bagging Decision Tree')

#     if cek_rasio_NB:
#         result_test_naive_bayes = clf_pf.predict([[0,	0,	0,	0,	0,	0,	1,	pendapatan_per_tahun,	durasi_peminjaman, jumlah_tanggungan]])[FIRST_IDX]
#         st.write(f"Customer Name : ", nama_nasabah,  "has risk rating", result_test_naive_bayes ,"based on Bagging Gaussian Naive Bayes model")
        
#     if cek_rasio_BNB:
#         result_test_naive_bayes_bagging = clf.predict([[0,	0,	0,	0,	0,	0,	1,	pendapatan_per_tahun,	durasi_peminjaman, jumlah_tanggungan]])[FIRST_IDX]
#         st.write(f"Customer Name : ", nama_nasabah,  "has risk rating", result_test_naive_bayes_bagging ,"based on Bagging Gaussian Naive Bayes model")

#     if cek_rasio_DC:
#         result_test_decision_tree = clf_tree.predict([[0,	0,	0,	0,	0,	0,	1,	pendapatan_per_tahun,	durasi_peminjaman, jumlah_tanggungan]])[FIRST_IDX]
#         st.write(f"Customer Name : ", nama_nasabah,  "has risk rating", result_test_decision_tree ,"based on Bagging Gaussian Naive Bayes model")
