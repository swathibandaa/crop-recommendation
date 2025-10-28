# web one

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Title
st.title("🌾 Smart Crop Recommendation Dashboard (KNN Classifier)")

# Load Dataset
@st.cache_data
def load_data():
    data = pd.read_csv("Crop_recommendation.csv")
    return data

data = load_data()

# Sidebar
st.sidebar.title("Input Soil and Weather Parameters")

# Input Fields
N = st.sidebar.slider("Nitrogen content (N)", 0, 140, 50)
P = st.sidebar.slider("Phosphorus content (P)", 5, 145, 50)
K = st.sidebar.slider("Potassium content (K)", 5, 205, 50)
temperature = st.sidebar.slider("Temperature (°C)", 10.0, 45.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 50.0)
ph = st.sidebar.slider("pH value", 3.0, 10.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

# Prepare Features
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Make prediction
input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                          columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
input_data_scaled = scaler.transform(input_data)
prediction = knn.predict(input_data_scaled)

# Display Prediction
st.success(f"✅ Recommended Crop for your farm: *{prediction[0]}*")

# Model Evaluation
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

st.write("### 📈 Model Accuracy")
st.metric(label="Accuracy", value=f"{accuracy:.2%}")

# Confusion Matrix
if st.checkbox("Show Confusion Matrix Heatmap"):
    cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
    plt.figure(figsize=(20, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=knn.classes_,
                yticklabels=knn.classes_)
    plt.xlabel('Predicted Crop')
    plt.ylabel('Actual Crop')
    plt.title('🌾 Confusion Matrix (KNN Classifier)')
    st.pyplot(plt)

# Data Preview
if st.checkbox("Show Raw Dataset"):

    st.write(data)
