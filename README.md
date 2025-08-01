# 🚢 Titanic Survival Prediction App

This project is a **creative and interactive web app** built using **Streamlit**, designed to predict whether a passenger would survive the Titanic disaster — based on historical data.

📍 **Live App**: [Try it here](https://titanic-prediction-model-fhvmadmny52hqzbskdz5rb.streamlit.app/)

---

## 🧠 About the Model

- The model is trained on the Titanic dataset with **logistic regression** from scratch.
- Accuracy: **~80%**
- Input features include:
  - Passenger Class
  - Fare Paid
  - Sex
  - Embarkation Port
  - Age Group
  - Family Size

---

## 🎨 Features of the Web App

- 🧠 Shows model accuracy and confidence score
- 🖼️ Titanic and RIP images based on prediction
- 🌊 Moving ocean-style background with dark theme
- 📱 Fully interactive with sidebar-based input
- ✅ Clean, responsive UI with styled markdown and buttons

---

## 🗂 Project Structure

```bash
.
├── app.py                 # Streamlit app
├── titanic_model.pkl      # Trained logistic regression model
├── scaler.pkl             # Feature scaler
├── features.pkl           # Selected feature columns
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
