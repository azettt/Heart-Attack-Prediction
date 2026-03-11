import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Load artifacts (pastikan file ini ada di folder artifacts/)
scaler = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")

def main():
    st.set_page_config(page_title="Heart Attack Prediction", layout="wide")
    
    st.title('Heart Attack Risk Prediction')
    st.write("Silakan masukkan data klinis pasien di bawah ini:")

    # Kita bagi jadi 2 kolom supaya UI lebih rapi
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age (Umur)', min_value=1, max_value=120, value=50)
        sex = st.selectbox('Sex (Jenis Kelamin)', options=[1, 0], format_func=lambda x: 'Male' if x == 1 else 'Female')
        cp = st.selectbox('Chest Pain Type (Tipe Nyeri Dada)', options=[0, 1, 2, 3])
        trestbps = st.number_input('Resting Blood Pressure (Tekanan Darah)', value=120)
        chol = st.number_input('Cholesterol (Kolesterol)', value=200)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[1, 0], format_func=lambda x: 'True' if x == 1 else 'False')
        restecg = st.selectbox('Resting ECG Results', options=[0, 1, 2])

    with col2:
        thalach = st.number_input('Max Heart Rate Achieved', value=150, help="Detak jantung maksimal yang dicapai saat tes.")
        exang = st.selectbox('Exercise Induced Angina', options=[1, 0], 
                             format_func=lambda x: 'Yes' if x == 1 else 'No',
                             help="Apakah dada terasa nyeri saat olahraga?")
        oldpeak = st.slider('Oldpeak (ST depression)', 0.0, 7.0, 1.0, step=0.1, 
                            help="Tingkat depresi grafik ST setelah olahraga dibanding saat istirahat.")
        slope = st.selectbox('Slope of Peak Exercise ST', options=[0, 1, 2],
                             help="0: Naik, 1: Datar, 2: Turun. (Indikasi kesehatan jantung saat stres)")
        ca = st.selectbox('Number of Major Vessels (0-4)', options=[0, 1, 2, 3, 4],
                          help="Jumlah pembuluh darah utama yang terlihat jelas melalui fluoroscopy.")
        thal = st.selectbox('Thalassemia', options=[0, 1, 2, 3],
                          help="Jenis kelainan darah (0: Normal, 1: Fixed Defect, 2: Reversible Defect, 3: DLL)")

    st.markdown("---")
    
    if st.button('Predict Risk'):
        # Susun fitur sesuai urutan kolom di dataset asli:
        # [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        
        prediction = make_prediction(features)
        
        if prediction == 1:
            st.error('⚠️ High Risk of Heart Attack')
        else:
            st.success('✅ Low Risk of Heart Attack')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    
    # 2. Transform pake scaler yang di-load
    X_scaled = scaler.transform(input_array)
    
    # 3. Predict
    prediction = model.predict(X_scaled)
    return prediction[0]

if __name__ == '__main__':

    main()
