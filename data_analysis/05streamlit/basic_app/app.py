import streamlit as st
import numpy as np
import pandas as pd
st.write("This is a simple text")


df = pd.DataFrame({
    'first column' : [1,2,3,4],
    'second colume': [1,4,9,16]
})

st.write("Here is the dataframe")
st.write(df)

chart_data = pd.DataFrame(
    np.random.randn(20,3),columns = ["a","b","c"]
)

st.line_chart(chart_data)