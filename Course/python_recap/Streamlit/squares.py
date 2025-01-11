import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title for the app
st.title("Numbers and Their Squares Visualization")

# Generate numbers and their squares
numbers = list(range(1, 101))
squares = [x**2 for x in numbers]

# Create a DataFrame
data = pd.DataFrame({"Number": numbers, "Square": squares})

# Display DataFrame
st.subheader("Numbers and Their Squares")
st.write(data)

# Plot using Matplotlib
st.subheader("Line Plot of Numbers and Their Squares")
fig, ax = plt.subplots()
ax.plot(data["Number"], data["Square"], color="blue", marker="o", linestyle="-")
ax.set_title("Number vs Square")
ax.set_xlabel("Number")
ax.set_ylabel("Square")
ax.grid(True)
st.pyplot(fig)

# Optional Bar Chart using Streamlit
st.subheader("Bar Chart of Numbers and Their Squares")
st.bar_chart(data.set_index("Number"))