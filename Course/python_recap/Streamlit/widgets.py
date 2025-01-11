import streamlit as st
import pandas as pd

st.title("Streamlit Text Input")

name = st.text_input("Enter you name: ")


age = st.slider("Select your age: ",0,100,20)

st.write("Your age is: ", age)

options = ["Python","C++","Javascript"]
lang = st.selectbox("Choose your most useful language: ",options)
st.write(f"You selected {lang}")

if name:
    st.write(f"Hello {name}")


