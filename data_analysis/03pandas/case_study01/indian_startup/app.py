import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide",page_title="StartUp Analysis",page_icon="image.png")

#reading cleaned csv from kaggle
df = pd.read_csv("startup_clean.csv")
df['date'] = pd.to_datetime(df['date'], dayfirst=True,errors = 'coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

st.sidebar.title("Indian Startups Analysis")

option = st.sidebar.selectbox("Select One" ,['Overall Analysis' , 'StartUp' , 'Investors'])

def load_overall_analysis():
    st.title("Overall Analysis")
    #Total amount invested
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        total = round(df['amount'].sum())
        st.metric("Total Amount Invested",str(total) + "Cr")
    #maximum funding recieved by startups
    with col2:
        max_fund = round(df.groupby("startup")['amount'].max().sort_values(ascending = False).head(1).values[0])
        st.metric("Max Amount Invested",str(max_fund) + "Cr")
    with col3:
        avg = round(df.groupby("startup")['amount'].sum().mean())
        st.metric("Avg. Amount Invested",str(avg) + "Cr")
    with col4:
        total_statups = df['startup'].nunique()
        st.metric("Total Funded Startups",str(total_statups))

    st.header("Month on Month Graph")
    options = st.selectbox("Select Analysis" , ["Count","Amount"])
    if options == "Count":
        temp_df = df.groupby(['year','month'])['startup'].count().reset_index()
        temp_df["x_axis"] = temp_df['month'].astype('str') + '-' + temp_df['year'].astype('str') 
        fig40 ,ax40 = plt.subplots()
        ax40.plot(temp_df['x_axis'],temp_df['startup'])
        st.pyplot(fig40)
    else:
        temp_df = df.groupby(['year','month'])['amount'].sum().reset_index()
        temp_df["x_axis"] = temp_df['month'].astype('str') + "-" + temp_df['year'].astype('str') 
        fig40 ,ax40 = plt.subplots()
        ax40.plot(temp_df['x_axis'],temp_df['amount'])
        st.pyplot(fig40)
      





def load_investor_detail(investor):
    st.title(investor)
    #recent 5 investment 
    last_5 = df[df['investors'].str.contains(investor)].head()[['date','startup','vertical','city','round','amount']]
    st.subheader("Most Recent Investment")
    st.dataframe(last_5)
    #Biggest Investment
   
    col1 , col2 = st.columns(2)
    with col1:
        big_invest = df[df['investors'].str.contains(investor)].groupby("startup")['amount'].sum().sort_values(ascending = False).head()
        st.subheader("Biggest Investment")
        fig ,ax = plt.subplots()
        ax.bar(big_invest.index,big_invest.values)
        st.pyplot(fig)
    with col2:
        verticals = df[df['investors'].str.contains(investor)].groupby("vertical")['amount'].sum()
        st.subheader("Sector Invested")
        fig1 ,ax1 = plt.subplots()
        ax1.pie(verticals,labels = verticals.index)
        st.pyplot(fig1)

    col3, col4 = st.columns(2)
    with col3:
        round = df[df['investors'].str.contains(investor)].groupby("round")['amount'].sum()
        st.subheader("Round Invested")
        fig2 ,ax2 = plt.subplots()
        ax2.pie(round,labels =round.index)
        st.pyplot(fig2)

    with col4:
       city = df[df['investors'].str.contains(investor)].groupby("city")['amount'].sum()
       st.subheader("City Invested")
       fig3 ,ax3 = plt.subplots()
       ax3.pie(city,labels = city.index)
       st.pyplot(fig3)


    df['year'] = df['date'].dt.year
    year_series = df[df['investors'].str.contains(investor)].groupby("year")['amount'].sum()
    st.subheader("Yearly Investment")
    fig4 ,ax4 = plt.subplots()
    ax4.plot(year_series.index,year_series.values)
    st.pyplot(fig4)
        

if option == "Overall Analysis":
    load_overall_analysis()
elif option == "StartUp":
    st.title("StartUp Analysis")
    st.sidebar.selectbox("Select StartUp" ,sorted(df['startup'].unique().tolist()))
    btn1 = st.sidebar.button("Find Startup Details")
else:
    selected_investor = st.sidebar.selectbox("Select Investor" ,sorted(set(df['investors'].str.split(",").sum())))
    btn2 = st.sidebar.button("Find Investor Details")
    if btn2:
        load_investor_detail(selected_investor)

