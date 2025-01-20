import streamlit as st

# Title
st.title("My First Streamlit App")

# Text
st.write("Hello, Streamlit!")

# Display a text input widget
name = st.text_input("Enter your name:")
st.write(f"Hello {name}!")

# Display a slider widget
age = st.slider("Select your age", 0, 100, 25)
st.write(f"Your age is {age}")
