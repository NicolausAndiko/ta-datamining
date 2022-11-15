import streamlit as st

st.write("# Form Biodata")

# st.markdown("Form Biodata")
form1 = st.form(key = "form1")
nama = form1.text_input(label = "Nama")
alamat = form1.text_input(label = "Alamat")
umur = form1.number_input("Umur", min_value = 0, max_value = 100)
status = form1.selectbox('Status Pekerjaan', ['Bekerja', 'Tidak Bekerja'], key=1)
btnSubmit = form1.form_submit_button(label = "Submit")

if (btnSubmit) :
    st.text(nama)
    st.text(alamat)
    st.text(umur)
    st.text(status)