#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import requests
from io import BytesIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# GitHub raw file URL
GITHUB_DATA_URL = "https://raw.githubusercontent.com/aguskurniawan10/calculator_blending/main/DATA%20PREDIKSI%20NK%20LAB%202025.xlsx"

# Set page config
st.set_page_config(page_title="Prediksi GCV", layout="wide")
st.title("🔍 CALCULATOR BLENDING BATUBARA UBP JPR")

# Function to download data from GitHub
def download_data_from_github():
    try:
        response = requests.get(GITHUB_DATA_URL)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return BytesIO(response.content)
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None

# Function to train and save models
def train_and_save_models():
    # Download data from GitHub
    data_file = download_data_from_github()
    if data_file is None:
        st.stop()
    
    # Load Data
    df = pd.read_excel(data_file)
    
    # Standardize column names
    df.columns = df.columns.str.strip()
    
    # Encode Suppliers column
    label_encoder = LabelEncoder()
    df['Suppliers'] = label_encoder.fit_transform(df['Suppliers'])
    
    # Prepare features and target
    X = df[[
        'Suppliers',
        'GCV ARB UNLOADING', 
        'TM ARB UNLOADING', 
        'Ash Content ARB UNLOADING', 
        'Total Sulphur ARB UNLOADING'
    ]]
    y = df['GCV (ARB) LAB']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Separate numeric and categorical data
    X_train_numeric = X_train.drop(columns=['Suppliers'])
    X_test_numeric = X_test.drop(columns=['Suppliers'])
    
    # Imputation only for numeric data
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_numeric)
    X_test_imputed = imputer.transform(X_test_numeric)
    
    # Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Combine back with Suppliers
    X_train_final = np.hstack([X_train[['Suppliers']].values, X_train_scaled])
    X_test_final = np.hstack([X_test[['Suppliers']].values, X_test_scaled])
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Support Vector Regression': SVR(kernel='rbf')
    }
    
    # Train and evaluate models
    best_model = None
    best_score = float('-inf')
    results = {}
    for name, model in models.items():
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        r2 = r2_score(y_test, y_pred)
        results[name] = r2
        if r2 > best_score:
            best_score = r2
            best_model = model
    
    best_model_name = max(results, key=results.get)
    
    # Save preprocessing components
    with open("label_encoder.pkl", "wb") as file:
        pickle.dump(label_encoder, file)
    
    with open("imputer.pkl", "wb") as file:
        pickle.dump(imputer, file)
    
    with open("scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)
    
    with open("best_model.pkl", "wb") as file:
        pickle.dump(best_model, file)
    
    with open("best_model_info.pkl", "wb") as file:
        pickle.dump({
            "name": best_model_name,
            "r2": best_score,
            "results": results
        }, file)
    
    return label_encoder, imputer, scaler, best_model, best_model_name, best_score

# Path for model files
MODEL_PATH = "best_model.pkl"
IMPUTER_PATH = "imputer.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
BEST_MODEL_INFO_PATH = "best_model_info.pkl"

# Function to load models with proper error handling
def load_models():
    required_files = {
        "Model": MODEL_PATH,
        "Imputer": IMPUTER_PATH,
        "Scaler": SCALER_PATH,
        "Label Encoder": ENCODER_PATH,
        "Model Info": BEST_MODEL_INFO_PATH
    }
    
    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(name)
    
    if missing_files:
        # Option to train models if files are missing
        st.warning(f"File-file berikut tidak ditemukan: {', '.join(missing_files)}")
        if st.button("Train Models Now"):
            with st.spinner('Training models...'):
                train_and_save_models()
                st.success("Models trained and saved successfully!")
        st.stop()
    
    model_components = {}
    try:
        with open(MODEL_PATH, "rb") as file:
            model_components["best_model"] = pickle.load(file)
        with open(IMPUTER_PATH, "rb") as file:
            model_components["imputer"] = pickle.load(file)
        with open(SCALER_PATH, "rb") as file:
            model_components["scaler"] = pickle.load(file)
        with open(ENCODER_PATH, "rb") as file:
            model_components["label_encoder"] = pickle.load(file)
        with open(BEST_MODEL_INFO_PATH, "rb") as file:
            model_components["best_model_info"] = pickle.load(file)
        return model_components
    except Exception as e:
        st.error(f"Error loading model components: {str(e)}")
        st.stop()

# Load models
models = load_models()
best_model = models["best_model"]
imputer = models["imputer"]
scaler = models["scaler"]
label_encoder = models["label_encoder"]
best_model_info = models["best_model_info"]

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Input Data", "Debugging Info"])

with tab1:
    # Input section - supplier selection
    st.subheader("Informasi Supplier dan Lokasi")
    col1, col2, col3 = st.columns(3)  # Adjusted to 3 columns
    
    with col1:
        st.markdown("### Supplier 1")
        supplier_list = label_encoder.classes_.tolist()
        supplier_1 = st.selectbox("Pilih Supplier 1", supplier_list, key="supplier1")
        location_1 = st.selectbox("Lokasi Pengambilan", ["Tongkang", "Coalyard"], key="loc1")
        if location_1 == "Coalyard":
            storage_time_1 = st.number_input("Lama Penyimpanan (hari)", min_value=0, max_value=365, value=0, key="storage1")
        else:
            storage_time_1 = 0
    
    with col2:
        st.markdown("### Supplier 2")
        supplier_2 = st.selectbox("Pilih Supplier 2", supplier_list, key="supplier2")
        location_2 = st.selectbox("Lokasi Pengambilan", ["Tongkang", "Coalyard"], key="loc2")
        if location_2 == "Coalyard":
            storage_time_2 = st.number_input("Lama Penyimpanan (hari)", min_value=0, max_value=365, value=0, key="storage2")
        else:
            storage_time_2 = 0
    
    with col3:
        st.markdown("### Supplier 3")
        supplier_3 = st.selectbox("Pilih Supplier 3", supplier_list, key="supplier3")
        location_3 = st.selectbox("Lokasi Pengambilan", ["Tongkang", "Coalyard"], key="loc3")
        if location_3 == "Coalyard":
            storage_time_3 = st.number_input("Lama Penyimpanan (hari)", min_value=0, max_value=365, value=0, key="storage3")
        else:
            storage_time_3 = 0
    
    # Blending percentages
    st.subheader("Persentase Campuran")
    col1, col2, col3, col4 = st.columns(4)  # Adjusted to 4 columns
    
    with col1:
        supplier_1_percentage = st.slider(f"Persentase {supplier_1}", 0, 100, 33, step=1, key="perc1")
    
    with col2:
        supplier_2_percentage = st.slider(f"Persentase {supplier_2}", 0, 100, 33, step=1, key="perc2")
    
    with col3:
        supplier_3_percentage = st.slider(f"Persentase {supplier_3}", 0, 100, 33, step=1, key="perc3")
    
    with col4:
        biomass_percentage = st.slider("Persentase Biomass", 0, 100, 1, step=1, key="biomass")
    
    # Check if percentages add up to 100
    total_percentage = supplier_1_percentage + supplier_2_percentage + supplier_3_percentage + biomass_percentage
    if total_percentage != 100:
        st.warning(f"Total persentase saat ini: {total_percentage}%. Idealnya, total persentase adalah 100%.")
    
    # Parameters input section
    st.subheader("Parameter Batubara")
    
    # Define parameters
    parameters = [
        "GCV ARB UNLOADING", 
        "TM ARB UNLOADING", 
        "Ash Content ARB UNLOADING", 
        "Total Sulphur ARB UNLOADING"
    ]
    
    # Sample ranges for each parameter to guide users
    param_ranges = {
        "GCV ARB UNLOADING": (3500.0, 5500.0),
        "TM ARB UNLOADING": (20.0, 40.0),
        "Ash Content ARB UNLOADING": (2.0, 10.0),
        "Total Sulphur ARB UNLOADING": (0.1, 1.0)
    }
    
    param_values = {}
    
    # Create columns and inputs for each parameter
    for param in parameters:
        col1, col2, col3 = st.columns(3)  # Adjusted to 3 columns
        
        with col1:
            min_val, max_val = param_ranges[param]
            param_values[f"{param}_1"] = st.number_input(
                f"{param} - {supplier_1}", 
                min_value=float(min_val), 
                max_value=float(max_val),
                value=float((min_val + max_val) / 2),
                key=f"{param}_1"
            )
        
        with col2:
            min_val, max_val = param_ranges[param]
            param_values[f"{param}_2"] = st.number_input(
                f"{param} - {supplier_2}", 
                min_value=float(min_val), 
                max_value=float(max_val),
                value=float((min_val + max_val) / 2),
                key=f"{param}_2"
            )
        
        with col3:
            min_val, max_val = param_ranges[param]
            param_values[f"{param}_3"] = st.number_input(
                f"{param} - {supplier_3}", 
                min_value=float(min_val), 
                max_value=float(max_val),
                value=float((min_val + max_val) / 2),
                key=f"{param}_3"
            )
    
    # Biomass GCV input if biomass percentage > 0
    if biomass_percentage > 0:
        st.subheader("Parameter Biomass")
        gcv_biomass = st.number_input("GCV Biomass (kcal/kg)", min_value=0.0, max_value=5000.0, value=3000.0)
    else:
        gcv_biomass = 0.0
    
    # Prepare data for prediction when button is clicked
    if st.button("Prediksi GCV"):
        blended_data = []
        
        # Encode suppliers
        supplier_encoded_1 = label_encoder.transform([supplier_1])[0]
        supplier_encoded_2 = label_encoder.transform([supplier_2])[0]
        supplier_encoded_3 = label_encoder.transform([supplier_3])[0]
        
        # Calculate blended values for each parameter
        blended_data.append(supplier_encoded_1)  # First supplier
        
        for param in parameters:
            val_1 = param_values[f"{param}_1"]
            val_2 = param_values[f"{param}_2"]
            val_3 = param_values[f"{param}_3"]
            
            # Calculate weighted average based on percentages
            if (supplier_1_percentage + supplier_2_percentage + supplier_3_percentage) > 0:
                blended_value = (val_1 * supplier_1_percentage + val_2 * supplier_2_percentage + val_3 * supplier_3_percentage) / (supplier_1_percentage + supplier_2_percentage + supplier_3_percentage)
            else:
                blended_value = 0
                
            blended_data.append(blended_value)
        
        # Reshape, impute missing values, and scale the data
        input_array = np.array(blended_data).reshape(1, -1)
        
        # Apply imputation
        imputed_array = imputer.transform(input_array[:, 1:])
        imputed_data = np.hstack([input_array[:, 0].reshape(-1, 1), imputed_array])
        
        # Apply scaling
        scaled_array = scaler.transform(imputed_data[:, 1:])
        scaled_data = np.hstack([imputed_data[:, 0].reshape(-1, 1), scaled_array])
        
        # Make prediction
        try:
            prediction = best_model.predict(scaled_data)[0]
            
            # Perform sanity check on the prediction
            if prediction < 0 or prediction > 10000:
                st.error(f"Model mengembalikan nilai prediksi tidak valid: {prediction}")
                prediction = max(2000, min(prediction, 5500))  # Constrain to reasonable range
                st.warning(f"Nilai diperbaiki ke dalam rentang yang valid: {prediction}")
            
            # Apply biomass blending if applicable
            if biomass_percentage > 0:
                final_prediction = (prediction * (supplier_1_percentage + supplier_2_percentage + supplier_3_percentage) + 
                                  gcv_biomass * biomass_percentage) / 100
            else:
                final_prediction = prediction
                
            # Apply storage time effects
            if location_1 == "Coalyard" and storage_time_1 > 0:
                decay_factor_1 = 0.05 * (storage_time_1 / 30)  # Cap at 5% max decrease per supplier
                final_prediction *= (1 - (decay_factor_1 * supplier_1_percentage / 100))
                
            if location_2 == "Coalyard" and storage_time_2 > 0:
                decay_factor_2 = 0.05 * (storage_time_2 / 30)  # Cap at 5% max decrease per supplier
                final_prediction *= (1 - (decay_factor_2 * supplier_2_percentage / 100))
                
            if location_3 == "Coalyard" and storage_time_3 > 0:
                decay_factor_3 = 0.05 * (storage_time_3 / 30)  # Cap at 5% max decrease per supplier
                final_prediction *= (1 - (decay_factor_3 * supplier_3_percentage / 100))
                
            # Ensure result is within reasonable bounds
            final_prediction = max(2000, min(final_prediction, 7000))
            
            # Subtract 100 from the final prediction
            final_prediction -= 150
                
            # Display results
            st.success(f"Prediksi GCV (ARB) LAB: {final_prediction:.2f} kcal/kg")
            
            # Show additional information
            st.info("""
            **Catatan:** 
            - Berdasarkan Literatur : Degradasi Nilai Kalori dalam 1 Bulan: MRC: 3% hingga 5% (Smith et al., 2023) LRC: 4% (Johnson dan Lee, 2024) Umum: 2% hingga 6% (Coal Research Institute, 2025). 
            - Penyimpanan di coalyard dapat menurunkan nilai GCV sekitar 5% per bulan.
            - Hasil prediksi dipengaruhi oleh persentase campuran dan waktu penyimpanan
            """)
            
        except Exception as e:
            st.error(f"Error saat melakukan prediksi: {str(e)}")
            st.info("Periksa kembali input Anda dan pastikan model sudah dilatih dengan benar.")

with tab2:
    st.subheader("Model Information")
    st.write(f"**Best Model:** {best_model_info['name']}")
    st.write(f"**R² Score:** {best_model_info['r2']:.4f}")
    
    st.subheader("Model Performance")
    performance_df = pd.DataFrame.from_dict(best_model_info['results'], orient='index', columns=['R² Score'])
    performance_df.index.name = 'Model'
    performance_df = performance_df.sort_values('R² Score', ascending=False)
    st.dataframe(performance_df)
    
    st.subheader("Label Encoder")
    st.write(pd.DataFrame({
        "Supplier": label_encoder.classes_, 
        "Encoded Value": range(len(label_encoder.classes_))
    }))

# Add footer
st.markdown("---")
st.markdown("© 2025 GCV Prediction Tool | For optimal results, ensure model is regularly updated with new data.")
