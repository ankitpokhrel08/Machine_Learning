import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to generate prime numbers up to a limit
def generate_primes(limit):
    primes = []
    for num in range(2, limit + 1):
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes

# Streamlit app title
st.title("Prime Numbers Visualization")

# User input for the range of prime numbers
st.sidebar.header("Prime Numbers Range")
max_limit = st.sidebar.slider("Select the upper limit", min_value=10, max_value=500, value=100)

# Generate prime numbers
prime_numbers = generate_primes(max_limit)

# Create a DataFrame
data = pd.DataFrame({"Index": range(1, len(prime_numbers) + 1), "Prime Number": prime_numbers})

# Display the DataFrame
st.subheader("Prime Numbers")
st.write(data)

# Dot graph using Matplotlib
st.subheader("Dot Graph of Prime Numbers")
fig, ax = plt.subplots()
ax.scatter(data["Index"], data["Prime Number"], color="red", s=50, label="Prime Numbers")
ax.set_title("Prime Numbers Visualization")
ax.set_xlabel("Index")
ax.set_ylabel("Prime Number")
ax.grid(True)
ax.legend()
st.pyplot(fig)