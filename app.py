import streamlit as st
import numpy as np
import pickle

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title of the app
st.title("Fake Profile Detector using ANN")

st.write("Enter profile details to check whether it is FAKE or REAL")

# Input fields
followers = st.number_input("Number of Followers", min_value=0)
following = st.number_input("Number of Following", min_value=0)
posts = st.number_input("Number of Posts", min_value=0)
bio_length = st.number_input("Bio Length", min_value=0)
profile_pic = st.selectbox("Profile Picture", ["Yes", "No"])

# Convert categorical to numeric
if profile_pic == "Yes":
    profile_pic = 1
else:
    profile_pic = 0

# Prediction button
if st.button("Check Profile"):

    # Create input array
    input_data = np.array([[followers, following, posts, bio_length, profile_pic]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Output result
    if prediction[0] == 1:
        st.error("This is a FAKE Profile ❌")
    else:
        st.success("This is a REAL Profile ✅")