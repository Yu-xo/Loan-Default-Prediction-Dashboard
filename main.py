import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- Utility Function for Imputation ---
def preprocess_data(df):
    """Handles missing values and categorical encoding."""
    
    # Drop any identifier columns that won't be useful for modeling (e.g., 'Loan_ID')
    df = df.drop(columns=[col for col in df.columns if 'id' in col.lower()], errors='ignore')

    # Separate data types for imputation
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include='object').columns

    # 1. Imputation (Handling Missing Values)
    
    # Fill numerical NaNs with the mean
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    
    # Fill categorical NaNs with the mode (most frequent value)
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 2. Encoding (Converting Strings to Numbers)
    
    # Apply One-Hot Encoding to all object columns
    # drop_first=True helps avoid multicollinearity
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df_encoded

# ------------------------------
# App Title
# ------------------------------
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("💳 Loan Default Prediction Dashboard")
st.markdown("This app cleans and encodes your dataset, trains Logistic Regression, Random Forest, and XGBoost models, and compares their performance.")

# ------------------------------
# Upload CSV
# ------------------------------
uploaded_file = st.file_uploader("Upload your loan dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 1. Dataset Preview")
    st.dataframe(df.head())

    # ------------------------------
    # Preprocessing (The Fix for the ValueError)
    # ------------------------------
    
    # Check for a 'Default' target column and create a demo one if missing
    if "Default" not in df.columns:
        st.warning("⚠️ 'Default' column not found. Creating a random 'Default' column for demonstration. Please ensure your dataset has a binary target variable.")
        df["Default"] = np.random.choice([0, 1], size=len(df))  # demo only

    # 1. Separate Target
    target_col = 'Default'
    
    # 2. Preprocess the features (Imputation and Encoding)
    df_encoded = preprocess_data(df.drop(columns=[target_col]))
    
    # Combine features and target for splitting
    X = df_encoded
    y = df[target_col].astype(int)

    st.write("### 2. Processed Features Preview (All Numerical)")
    st.dataframe(X.head())
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.success(f"Data successfully preprocessed. X_train shape: {X_train.shape}")


    # ------------------------------
    # Train Models
    # ------------------------------
    st.write("### 3. Training Models...")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        # Removed use_label_encoder=False as it is now deprecated, relying on pd.get_dummies
        "XGBoost": XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric="logloss", random_state=42)
    }

    results = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            roc = roc_auc_score(y_test, y_proba)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            results[name] = {"model": model, "roc": roc, "cm": cm, "report": report}
            st.success(f"{name} trained successfully.")
            
        except Exception as e:
            st.error(f"Error training {name}: {e}")
            
    # ------------------------------
    # Display Results
    # ------------------------------
    st.write("## 📊 4. Model Comparison")

    model_names = list(results.keys())
    if model_names:
        cols = st.columns(len(model_names))
        
        for i, name in enumerate(model_names):
            res = results[name]
            with cols[i]:
                st.subheader(name)
                st.metric(label="ROC-AUC Score", value=f"{res['roc']:.4f}")

                # Confusion Matrix
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(res["cm"], annot=True, fmt="d", cmap="viridis", ax=ax, cbar=False,
                            xticklabels=['No Default (0)', 'Default (1)'],
                            yticklabels=['Actual 0', 'Actual 1'])
                ax.set_title("Confusion Matrix")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)

                # Classification Report Summary
                st.write("**Classification Report:**")
                report_df = pd.DataFrame(res["report"]).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0, subset=['precision', 'recall', 'f1-score'], color='lightgreen'))

    # ------------------------------
    # New Applicant Prediction
    # ------------------------------
    st.write("## 🧑‍💼 5. Predict for New Applicant (XGBoost)")

    if "XGBoost" in results:
        prediction_model = results["XGBoost"]["model"]
        
        # We need the original, non-encoded columns for the user inputs
        original_features = df.drop(columns=[target_col, *[col for col in df.columns if 'id' in col.lower()]], errors='ignore')
        
        input_data = {}
        input_cols = st.columns(3)
        
        for i, col in enumerate(original_features.columns):
            with input_cols[i % 3]:
                # Check data type of the original column
                col_dtype = original_features[col].dtype
                
                if col_dtype == object:
                    # Input for categorical features (Dropdown/Selectbox)
                    options = original_features[col].unique().tolist()
                    input_data[col] = st.selectbox(f"Select {col}", options)
                else:
                    # Input for numerical features (Number Input)
                    min_val = float(original_features[col].min())
                    max_val = float(original_features[col].max())
                    mean_val = float(original_features[col].mean())
                    input_data[col] = st.number_input(f"Enter {col}", min_value=min_val, max_value=max_val, value=mean_val)

        if st.button("Calculate Prediction"):
            # Create a DataFrame from user input
            input_df = pd.DataFrame([input_data])
            
            # Re-apply the same preprocessing steps to the input data
            
            # 1. Imputation (not strictly needed since all fields are filled, but good practice)
            # 2. Encoding
            
            # Get original categorical columns
            orig_categorical_cols = original_features.select_dtypes(include='object').columns.tolist()

            # Apply One-Hot Encoding to the input
            input_df_encoded = pd.get_dummies(input_df, columns=orig_categorical_cols, drop_first=True)
            
            # Align columns: This is CRITICAL. The prediction columns must match X_train columns exactly.
            # Create a zero vector matching X_train columns
            final_input = pd.DataFrame(0, index=[0], columns=X_train.columns)
            
            # Fill in the values from the encoded input
            for col in input_df_encoded.columns:
                if col in final_input.columns:
                    final_input[col] = input_df_encoded[col].iloc[0]

            # Predict
            pred = prediction_model.predict(final_input)[0]
            proba = prediction_model.predict_proba(final_input)[0][1]

            st.markdown(f"""
            ### Prediction Result
            <div style='padding: 15px; border-radius: 10px; background-color: #e6f7ff; border: 1px solid #91d5ff; color: #00508f;'>
                <p style='font-size: 1.2em; margin-bottom: 5px;'>
                    Prediction: <strong>{'DEFAULT' if pred == 1 else 'NO DEFAULT'}</strong>
                </p>
                <p style='font-size: 1.5em; font-weight: bold; margin-bottom: 0;'>
                    Risk Probability: <span style='color: {'#ff4d4f' if proba > 0.5 else '#52c41a'};'>{proba:.2f}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.error("XGBoost model could not be trained. Please check the dataset for issues.")
