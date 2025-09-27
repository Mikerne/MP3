import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# --- Load models and encoders ---
rf_model = joblib.load("../models/rf_attrition_model.joblib")
ridge_model = joblib.load("../models/ridge_monthlyincome_model.joblib")
kmeans_model = joblib.load("../models/kmeans_clusters.joblib")
scaler = joblib.load("../models/scaler_for_clustering.joblib")
classification_encoder = joblib.load("../models/classification_encoder.joblib")  # gem en encoder fra preprocessing

# --- Load dataset for visualization ---
df = pd.read_csv("../data/processed_hr_dataset.csv")

st.title("Employee Behavior Prediction App")
st.write("Predict employee behavior or explore the HR dataset.")
scaler_reg = joblib.load("../models/scaler_for_regression.joblib")

# --- Sidebar for model selection ---
model_choice = st.sidebar.selectbox("Select Model", 
                                    ["Attrition Classification", "Monthly Income Regression", "Employee Clustering", "Explore Data"])

# --- Function: user input for classification/regression ---
# --- User input for regression ---
def user_input_regression():
    st.subheader("Provide employee information for Monthly Income Prediction")
    
    # --- Numeric features (excl. MonthlyIncome) ---
    numeric_features = ["Age", "DailyRate", "DistanceFromHome", "YearsAtCompany",
                        "YearsInCurrentRole", "TotalWorkingYears", "JobLevel",
                        "StockOptionLevel", "TrainingTimesLastYear", "NumCompaniesWorked"]
    
    input_data = {}
    for feature in numeric_features:
        input_data[feature] = st.number_input(feature, value=int(df[feature].mean()))
    
    # --- Categorical features ---
    job_roles = ["Healthcare Representative", "Human Resources", "Laboratory Technician",
                 "Manager", "Manufacturing Director", "Research Director", "Research Scientist",
                 "Sales Executive", "Sales Representative"]
    departments = ["Human Resources", "Research & Development", "Sales"]
    genders = ["Male", "Female"]
    marital_statuses = ["Single", "Married", "Divorced"]
    business_travel = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    overtime = ["Yes", "No"]
    education_fields = ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]

    input_data["JobRole"] = st.selectbox("JobRole", job_roles)
    input_data["Department"] = st.selectbox("Department", departments)
    input_data["Gender"] = st.selectbox("Gender", genders)
    input_data["MaritalStatus"] = st.selectbox("MaritalStatus", marital_statuses)
    input_data["BusinessTravel"] = st.selectbox("BusinessTravel", business_travel)
    input_data["OverTime"] = st.selectbox("OverTime", overtime)
    input_data["EducationField"] = st.selectbox("EducationField", education_fields)

    # --- Convert to dataframe ---
    df_input = pd.DataFrame([input_data])

    # --- Encode categorical features ---
    categorical_features = ["JobRole", "Department", "EducationField", "MaritalStatus",
                            "Gender", "BusinessTravel", "OverTime"]
    encoded = classification_encoder.transform(df_input[categorical_features])
    encoded_df = pd.DataFrame(encoded, columns=classification_encoder.get_feature_names_out(categorical_features))

    # --- Scale numeric features for regression ---
    df_input[numeric_features] = scaler_reg.transform(df_input[numeric_features])

    # --- Combine numeric + encoded categoricals ---
    df_input_final = pd.concat([df_input[numeric_features], encoded_df], axis=1)
    return df_input_final


def user_input_classification():
    st.subheader("Provide employee information for Attrition Prediction")
    
    # --- Numeric features ---
    numeric_features = ["Age", "DailyRate", "DistanceFromHome", "MonthlyIncome", "YearsAtCompany",
                        "YearsInCurrentRole", "TotalWorkingYears", "JobLevel",
                        "StockOptionLevel", "TrainingTimesLastYear", "NumCompaniesWorked"]
    
    input_data = {}
    for feature in numeric_features:
        input_data[feature] = st.number_input(feature, value=int(df[feature].mean()))
    
    # --- Categorical features ---
    job_roles = ["Healthcare Representative", "Human Resources", "Laboratory Technician",
                 "Manager", "Manufacturing Director", "Research Director", "Research Scientist",
                 "Sales Executive", "Sales Representative"]
    departments = ["Human Resources", "Research & Development", "Sales"]
    genders = ["Male", "Female"]
    marital_statuses = ["Single", "Married", "Divorced"]
    business_travel = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    overtime = ["Yes", "No"]
    education_fields = ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]

    input_data["JobRole"] = st.selectbox("JobRole", job_roles)
    input_data["Department"] = st.selectbox("Department", departments)
    input_data["Gender"] = st.selectbox("Gender", genders)
    input_data["MaritalStatus"] = st.selectbox("MaritalStatus", marital_statuses)
    input_data["BusinessTravel"] = st.selectbox("BusinessTravel", business_travel)
    input_data["OverTime"] = st.selectbox("OverTime", overtime)
    input_data["EducationField"] = st.selectbox("EducationField", education_fields)

    # --- Convert to dataframe and encode categoricals ---
    df_input = pd.DataFrame([input_data])
    categorical_features = ["JobRole", "Department", "EducationField", "MaritalStatus",
                            "Gender", "BusinessTravel", "OverTime"]
    encoded = classification_encoder.transform(df_input[categorical_features])
    encoded_df = pd.DataFrame(encoded, columns=classification_encoder.get_feature_names_out(categorical_features))

    # --- Combine numeric + encoded categoricals ---
    df_input_final = pd.concat([df_input[numeric_features], encoded_df], axis=1)
    return df_input_final

# --- Classification ---
if model_choice == "Attrition Classification":
    st.header("Attrition Prediction")
    df_input = user_input_classification()
    prediction = rf_model.predict(df_input)
    prob = rf_model.predict_proba(df_input)
    st.success(f"Predicted Attrition: {'Yes' if prediction[0]==1 else 'No'}")
    st.info(f"Probability: No: {prob[0][0]:.2f}, Yes: {prob[0][1]:.2f}")

# --- Regression ---
elif model_choice == "Monthly Income Regression":
    st.header("Monthly Income Prediction")
    df_input = user_input_regression()  # korrekt funktion til regression
    prediction = ridge_model.predict(df_input)
    st.success(f"Predicted Monthly Income: ${prediction[0]:.2f}")

# --- Clustering ---
elif model_choice == "Employee Clustering":
    st.header("Employee Clustering")
    st.subheader("Provide numeric information for clustering:")
    cluster_features = ["Age", "DailyRate", "DistanceFromHome", "YearsAtCompany",
                        "YearsInCurrentRole", "TotalWorkingYears", "JobLevel",
                        "StockOptionLevel", "TrainingTimesLastYear", "NumCompaniesWorked",
                        "MonthlyIncome"]
    input_data = {}
    for feature in cluster_features:
        input_data[feature] = st.number_input(feature, value=int(df[feature].mean()))
    df_input = pd.DataFrame([input_data])
    df_scaled = scaler.transform(df_input)
    cluster_label = kmeans_model.predict(df_scaled)
    st.success(f"This employee belongs to Cluster: {cluster_label[0]}")

# --- Explore data ---
elif model_choice == "Explore Data":
    st.header("Explore HR Dataset")
    st.dataframe(df.head(20))
    st.subheader("Basic statistics")
    st.write(df.describe().T)
    
    st.subheader("Attrition distribution")
    st.bar_chart(df["Attrition"] if "Attrition" in df.columns else None)
    
    st.subheader("Monthly Income distribution")
    st.bar_chart(df["MonthlyIncome"])
