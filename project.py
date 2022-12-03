import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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

    url = "social_network_ads.csv"
    data = pd.read_csv(url)
    st.dataframe(data)

with preporcessing:
    st.write("""# Preprocessing""")
    st.write("Preprocessing adalah proses menyiapkan data dasar atau inti sebelum melakukan proses lainnya. \nPada dasarnya data preprocessing dapat dilakukan dengan membuang data yang tidak sesuai atau mengubah data menjadi bentuk yang lebih mudah untuk diproses oleh sistem. \nProses pembersihan meliputi penghilangan duplikasi data, pengisian atau penghapusan data yang hilang, pembetulan data yang tidak konsisten, dan pembetulan salah ketik. \nSeperti namanya, normalisasi dapat diartikan secara sederhana sebagai proses menormalkan data dari hal-hal yang tidak sesuai.")
    st.write("Dataset Asli:")
    st.dataframe(data)

    st.write("Drop User ID karena bukan merupakan fitur.")
    X = data.drop(columns=['User ID'])
    st.dataframe(X)

    st.write("Ubah data 'Gender' dari kategorikal menjadi numerik")
    split_overdue_X = pd.get_dummies(X["Gender"], prefix="Gender")
    X = X.join(split_overdue_X)
    X = X.drop(columns="Gender")
    st.dataframe(X)

    st.write("Pisahkan Purchased karena ini merupakan target")
    y = X.iloc[:,2].values
    X = X.drop(columns="Purchased")
    target_label = ["Purchased"]
    y = pd.DataFrame(y, columns=target_label)
    st.dataframe(X)
    st.dataframe(y)

    st.write("Data Setelah MinMaxScaler")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    features_names = ["Age", "Salary", "Gender_Female", "Gender_Male"]
    scaled_features = pd.DataFrame(scaled, columns=features_names)
    st.dataframe(scaled_features)

with modeling:
    st.write("# Modeling") 
    knn, naive_bayes, decision_tree, bagging_decision_tree = st.tabs(["K-NN","Naive Bayes", "Decision Tree", "Bagging Decision Tree"])

    X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.3, random_state=1)

    with knn:
        st.write("## K-Nearest Neighbor")
        st.write("Klasifikasi dengan menggunakan KNN dengan K = 100")
        k_range = (100)
        K = {}
        for k in range(1, k_range+1):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            Y_pred_knn=knn.predict(X_test)
            K[k] = accuracy_score(y_test, Y_pred_knn)
        
        knn_accuracy = round(100 * accuracy_score(y_test, Y_pred_knn), 2)
        best_k = max(K, key = K.get)
        knn_model = KNeighborsClassifier(n_neighbors=best_k)
        knn_model.fit(X_train, y_train)
        st.write("K terbaik = ",best_k-1,"dengan akurasi :",{max(K.values())* 100},"%")

    with naive_bayes:
        st.write("## Naive Bayes")
        naive_bayes_classifier = GaussianNB()
        naive_bayes_classifier.fit(X_train, y_train)
        Y_pred_nb = naive_bayes_classifier.predict(X_test)

        naive_bayes_accuracy = round(100 * accuracy_score(y_test, Y_pred_nb), 2)

        st.write('Akurasi dengan menggunakan Naive Bayes: {}%.'.format(naive_bayes_accuracy))

    with decision_tree:
        st.write("## Decision Tree")
        dt = DecisionTreeClassifier(criterion='gini')
        dt = dt.fit(X_train, y_train)
        Y_pred_dt = dt.predict(X_test)
        decision_tree_accuracy = round(100 * metrics.accuracy_score(y_test, Y_pred_dt), 2)

        st.write("Akurasi dengan model Decision Tree: {}%".format(decision_tree_accuracy))
        
    with bagging_decision_tree:
        st.write("## Bagging Decision Tree")
        clf_tree = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=10, random_state=0).fit(X_train, y_train)
        rsc = clf_tree.predict(X_test)
        c = ['Decision Tree']
        bag_tree = pd.DataFrame(rsc,columns = c)

        bagging_tree = round(100 * accuracy_score(y_test, bag_tree), 2)
        st.write('Akurasi dengan menggunakan Bagging Decision Tree: {} %.'.format(bagging_tree))


with implementation:
    st.write("# Implementation")
    imp_gender = st.selectbox('Gender',['Male','Female'])
    imp_age = st.slider('Age:', 0, 100)
    imp_salary = st.slider('Estimated Salary in USD (Year)', 15000, 150000)

    clicked = st.button("Predict Here")
    
    if clicked == True:
        knn, naive_bayes, decision_tree, bagging_decision_tree = st.tabs(["K-NN","Naive Bayes", "Decision Tree", "Bagging Decision Tree"])
        if imp_gender == "Male":
            imp_gender_female = 0
            imp_gender_male = 1
        elif imp_gender == "Female":
            imp_gender_female = 1
            imp_gender_male = 0
        
        data_imp = np.array([[imp_age, imp_salary, imp_gender_female, imp_gender_male]])
        X = X.append(pd.DataFrame(data_imp, columns = list(X)), ignore_index = True)
        newScaled = scaler.fit_transform(X)
        newScaled = pd.DataFrame(newScaled, columns=features_names)
        data_imp_scaler = newScaled.iloc[-1:]

        with knn:
            st.write("## Prediksi menggunakan K-NN")
            y_pred_knn_imp = knn_model.predict(data_imp_scaler)
            if (y_pred_knn_imp[0] == 0):
                st.write("Anda diprediksi tidak membeli barang ini")
            elif (y_pred_knn_imp[0] == 1):
                st.write("Anda diprediksi membeli barang ini")
        
        with naive_bayes:
            st.write("## Prediksi menggunakan Naive Bayes")
            y_pred_nb_imp = naive_bayes_classifier.predict(data_imp_scaler)
            if (y_pred_nb_imp[0] == 0):
                st.write("Anda diprediksi tidak membeli barang ini")
            elif (y_pred_nb_imp[0] == 1):
                st.write("Anda diprediksi membeli barang ini")

        with decision_tree:
            st.write("## Prediksi menggunakan Decision Tree")
            y_pred_dt_imp = dt.predict(data_imp_scaler)
            if (y_pred_dt_imp[0] == 0):
                st.write("Anda diprediksi tidak membeli barang ini")
            elif (y_pred_dt_imp[0] == 1):
                st.write("Anda diprediksi membeli barang ini")
        
        with bagging_decision_tree:
            st.write("## Prediksi menggunakan Bagging Decision Tree")
            y_pred_bag_imp = clf_tree.predict(data_imp_scaler)
            if (y_pred_bag_imp[0] == 0):
                st.write("Anda diprediksi tidak membeli barang ini")
            elif (y_pred_bag_imp[0] == 1):
                st.write("Anda diprediksi membeli barang ini")