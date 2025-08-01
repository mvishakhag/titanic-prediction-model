import streamlit as st
import numpy as np
import joblib
st.markdown(
    """
    <style>
    body {
        background-image: url("https://www.pixelstalk.net/wp-content/uploads/2016/06/HD-Underwater-Backgrounds-For-Desktop.jpg");  /* Replace with your image URL */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    .stApp {
        background-color: rgba(0, 0, 0, 0); /* Transparent to show the image */
    }

    h1, h2, h3, h4, h5, h6, p, div {
        color: white !important;
    }

    .css-18e3th9 {
        background-color: rgba(0, 0, 0, 0.6) !important;  /* Semi-transparent container */
        border-radius: 15px;
        padding: 10px;
    }
     </style>
    """,
    unsafe_allow_html=True
)
st.markdown("""
    <div style='
        background-color: rgba(0, 0, 0, 0.6);
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-size: 22px;
        font-family: Arial, sans-serif;
        margin-bottom: 30px;
    '>
        <h2 style='text-align:center;'>🚢 Welcome aboard!</h2>
        <p style='text-align:center;'>Let’s see if you’d survive the Titanic 🌊</p>
        <p style='text-align:center;'>Fill in your details on the sidebar 👈 and click <b>Predict</b>.</p>
    </div>
""", unsafe_allow_html=True)
# Load model
model = joblib.load("titanic_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True)
st.title("🚢 Titanic Survival Prediction")
accuracy = 0.8725
st.markdown(f"### 🎯 Model Accuracy: `{accuracy * 100:.2f}%`")


# Sidebar Inputs
st.sidebar.title("📝 Passenger Details")
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
fare = st.sidebar.slider("Fare Paid", 0.0, 100.0, 15.0)
sex_male = st.sidebar.selectbox("Sex", ["Male", "Female"])
embarked = st.sidebar.selectbox("Embarked Port", ["S", "Q"])
age_group = st.sidebar.selectbox("Age Group", ["0: Child", "1: Young Adult", "2: Middle Age", "3: Senior"])

# Transform input
input_data = np.array([[  
    pclass,
    fare,
    1 if sex_male == "Male" else 0,
    1 if embarked == "Q" else 0,
    1 if embarked == "S" else 0,
    int(age_group.split(":")[0])
]])

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

# Predict Button
if st.button("🔍 Predict"):
    proba = model.predict_proba(input_scaled)[0]
    survival_chance = proba[1] * 100
    death_chance = proba[0] * 100

    if prediction == 1:
        st.success("🎉 Prediction: You Would Survive!")
        st.metric("✅ Confidence", f"{survival_chance:.2f}%", delta=None)
    else:
        st.error("💀 Prediction: You Would Not Survive (like Jack 🥶)")
        st.image("https://th.bing.com/th/id/OIP.St9Lfor9ftp6HHpZaD-NRwHaEG?w=281&h=180&c=7&r=0&o=7&pid=1.7&rm=3", width=200, caption="RIP")
        st.metric("❌ Confidence", f"{death_chance:.2f}%", delta=None)
