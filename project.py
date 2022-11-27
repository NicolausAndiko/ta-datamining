import warnings
warnings.filterwarnings("ignore")

import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# # , LabelEncoder, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split
# from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC

from io import StringIO, BytesIO
import urllib.request
import joblib
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import os,sys
from scipy import stats

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
    # url = pd.read_csv("social_network_ads.csv", error_bad_lines=False)
    # st.dataframe(data)

    url = "social_network_ads.csv"
    data = pd.read_csv(url)
    df = pd.DataFrame(data)

with preporcessing:
    st.write("""# Preprocessing""")
    st.write("Preprocessing adalah proses menyiapkan data dasar atau inti sebelum melakukan proses lainnya. \nPada dasarnya data preprocessing dapat dilakukan dengan membuang data yang tidak sesuai atau mengubah data menjadi bentuk yang lebih mudah untuk diproses oleh sistem. \nProses pembersihan meliputi penghilangan duplikasi data, pengisian atau penghapusan data yang hilang, pembetulan data yang tidak konsisten, dan pembetulan salah ketik. \nSeperti namanya, normalisasi dapat diartikan secara sederhana sebagai proses menormalkan data dari hal-hal yang tidak sesuai.")
    st.write("Dataset Asli:")
    st.dataframe(data)

    st.write("Drop User ID karena bukan merupakan fitur.")
    Z = df.drop(columns=['User ID'])
    X = Z.drop(columns=['Purchased'])

    
    # st.write("Karena kolom Purchased adalah target, maka kita pisahkan variabel bebas dan variabel terikat")
    # X = data.iloc[:,:3].values
    # y = data.iloc[:,3].values
    # st.dataframe(X)
    # st.dataframe(y)

    #tramnsform
    X = pd.DataFrame(X)
    X['Gender'] = X['Gender'].astype('category')
    cat_columns = X.select_dtypes(['category']).columns
    X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)

    # st.write("Ubah data 'Gender' dari kategorikal menjadi numerik")
    # ct = ColumnTransformer([("Gender", OneHotEncoder(), [0])], remainder='passthrough')
    # X = ct.fit_transform(X)
    # st.dataframe(X)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)
    scaled_features


with modeling:
    st.write("""# Modeling""") 
    K_Nearest_Naighbour, naive_bayes, bagging_naive_bayes, bagging_decision_tree = st.tabs(["K-Nearest Neightbour", "Naive Bayes", "Bagging Naive Bayes", "Bagging Decision Tree"])

    y = data['EstimatedSalary'].values

    # encoder label
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    y_baru = le.fit_transform(y)

    # split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test=train_test_split(X, y_baru, test_size=0.2, random_state=1)

    # inisialisasi knn
    my_param_grid = {'n_neighbors':[2,3,5,7], 'weights': ['distance', 'uniform']}
    GridSearchCV(estimator=KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)
    knn = GridSearchCV(KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)
    knn.fit(X_train, y_train)

    pred_test = knn.predict(X_test)

    vknn = f'Hasil akurasi dari pemodelan K-Nearest Neighbour : {accuracy_score(y_test, pred_test) * 100 :.2f} %'

    filenameModelKnnNorm = 'modelKnnNorm.pkl'
    joblib.dump(knn, filenameModelKnnNorm)

    # inisialisasi model gausian
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    filenameModelGau = 'modelGau.pkl'
    joblib.dump(gnb, filenameModelGau)

    y_pred = gnb.predict(X_test)

    vg = f'Hasil akurasi dari pemodelan Gausian : {accuracy_score(y_test, y_pred) * 100 :.2f} %'

    # inisialisasi model decision tree
    from sklearn.tree import DecisionTreeClassifier
    d3 = DecisionTreeClassifier()
    d3.fit(X_train, y_train)

    filenameModeld3 = 'modeld3.pkl'
    joblib.dump(d3, filenameModeld3)

    y_pred = d3.predict(X_test)

    vd3 = f'Hasil akurasi dari pemodelan decision tree : {accuracy_score(y_test, y_pred) * 100 :.2f} %'

    # inisialisasi model k-mean
    from sklearn.cluster import KMeans
    km = KMeans()
    km.fit(X_train, y_train)

    filenameModelkm = 'modelkm.pkl'
    joblib.dump(km, filenameModelkm)

    y_pred = km.predict(X_test)

    vkm = f'Hasil akurasi dari pemodelan k-means clustering : {accuracy_score(y_test, y_pred) * 100 :.2f} %'


    with K_Nearest_Naighbour:
        st.header("K-Nearest Neighbour")
        st.write("Algoritma KNN mengasumsikan bahwa sesuatu yang mirip akan ada dalam jarak yang berdekatan atau bertetangga. Artinya data-data yang cenderung serupa akan dekat satu sama lain. KNN menggunakan semua data yang tersedia dan mengklasifikasikan data atau kasus baru berdasarkan ukuran kesamaan atau fungsi jarak. Data baru kemudian ditugaskan ke kelas tempat sebagian besar data tetangga berada.")
        
        st.header("Hasil Akurasi")
        st.write(vknn)


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
#         # model_accuracy['Gaussian Naive Bayes'] = naive_bayes_accuracy

#         naive_bayes_precision = round(100 * precision_score(y_test, Y_pred_nb, average = 'weighted'), 2)
#         # model_precision['Gaussian Naive Bayes'] = naive_bayes_precision

#         naive_bayes_recall = round(100 * recall_score(y_test, Y_pred_nb, average = 'weighted'), 2)
#         # model_recall['Gaussian Naive Bayes'] = naive_bayes_recall

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

# # with implementation:
# #     st.write("# Implementation")
# #     nama_nasabah = st.text_input('Masukkan Nama Nasabah')
# #     pendapatan_per_tahun = st.number_input('Masukkan pendapatan pertahun')
# #     durasi_peminjaman = st.number_input('Masukkan Durasi Peminjaman')
# #     jumlah_tanggungan = st.number_input('Masukkan Jumlah Tanggungan')

# #     clf = GaussianNB()
# #     clf.fit(matrices_X, matrices_Y)
# #     clf_pf = GaussianNB()
# #     clf_pf.partial_fit(matrices_X, matrices_Y, np.unique(matrices_Y))

# #     cek_rasio_NB = st.button('Cek Risk Ratio dengan Naive Bayes')
# #     cek_rasio_BNB = st.button('Cek Risk Ratio dengan Bagging Naive Bayes')
# #     cek_rasio_DC = st.button('Cek Risk Ratio dengan Bagging Decision Tree')

# #     if cek_rasio_NB:
# #         result_test_naive_bayes = clf_pf.predict([[0,	0,	0,	0,	0,	0,	1,	pendapatan_per_tahun,	durasi_peminjaman, jumlah_tanggungan]])[FIRST_IDX]
# #         st.write(f"Customer Name : ", nama_nasabah,  "has risk rating", result_test_naive_bayes ,"based on Bagging Gaussian Naive Bayes model")
        
# #     if cek_rasio_BNB:
# #         result_test_naive_bayes_bagging = clf.predict([[0,	0,	0,	0,	0,	0,	1,	pendapatan_per_tahun,	durasi_peminjaman, jumlah_tanggungan]])[FIRST_IDX]
# #         st.write(f"Customer Name : ", nama_nasabah,  "has risk rating", result_test_naive_bayes_bagging ,"based on Bagging Gaussian Naive Bayes model")

# #     if cek_rasio_DC:
# #         result_test_decision_tree = clf_tree.predict([[0,	0,	0,	0,	0,	0,	1,	pendapatan_per_tahun,	durasi_peminjaman, jumlah_tanggungan]])[FIRST_IDX]
# #         st.write(f"Customer Name : ", nama_nasabah,  "has risk rating", result_test_decision_tree ,"based on Bagging Gaussian Naive Bayes model")
