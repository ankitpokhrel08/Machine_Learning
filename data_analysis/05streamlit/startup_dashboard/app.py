import streamlit as st
import pandas as pd
import numpy as np
import time

st.title("Startup Dashboard")
st.header("Tech Startup")
st.subheader("Hello World!!!!")
st.write("lorem epsum hello  world I am ankit")
st.markdown("""
## This is right way to render markdowns in screens:
- This is first bullet points
- This is second one 
- This is third"""
            )

st.code("""
def sum(a,b):
        return a+b
sum(2,3)
""")

df = pd.read_csv('ipl-matches.csv')
newdf = df.head(5)
st.dataframe(newdf)
money = '3 Cr'
col1, col2,col3,col4,col5 = st.columns(5)
with col1:
    st.metric("Today Earning",money,'3%')
with col2:
    st.metric("Today Earning",money,'-3%')
with col3:
    st.metric("Today Earning",money,'3%')
with col4:
    st.metric("Today Earning",money,'-3%')
with col5:
    st.metric("Today Earning",money,'3%')


# st.image('image.png')

st.sidebar.title("This is sidebar")

# st.title("Loading Image...")
# bar = st.progress(0)

# for i in range(1,101):
#     time.sleep(0.1)
#     bar.progress(i)

# st.image('image.png')

# name = st.text_input("What is your name?")
# age =  st.number_input("How old are you?")
# date = st.date_input("Enter your DOB")

email = st.text_input("Enter Email")
password = st.text_input("Enter Password")
gender = st.selectbox('Select Gender' , ['Male',"Female","Other"])

btn = st.button("Login")

if btn:
    if email == "pokhrel@gmail.com" and password == "1234":
        st.success("LoggedIn Successful")
        st.balloons()
    else:
        st.error("LoggedIn Failed")


file = st.file_uploader("Upload your CSV file")
if file is not None:
    df = pd.read_csv(file)
    st.dataframe(df)





