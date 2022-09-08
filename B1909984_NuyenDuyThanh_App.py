from pickle import NONE
import streamlit as st
import pickle


st.title("Predict the iris flower family!")
with st.form("Form1"):
    sepal_length = st.number_input("sepal length :",min_value=0.0,max_value= 10.0)
    sepal_width = st.number_input("sepal width :",min_value=0.0,max_value= 10.0)
    petal_length = st.number_input("petal length :",min_value=0.0,max_value= 10.0)
    petal_width = st.number_input("petal width :",min_value=0.0,max_value= 10.0)
    submit = st.form_submit_button("Predict")
 

loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
predict = loaded_model.predict([[sepal_length,sepal_width,petal_length,petal_width]])

st.header("Variety: ")
st.write(predict)
