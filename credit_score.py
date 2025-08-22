# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ==============================
# Load & preprocess dataset
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("D:\\projects\\sid\\siddd.venv\\loan_approval_dataset.csv")
    df.columns = df.columns.str.strip()
    df.drop("loan_id", axis=1, inplace=True)

    # Encode categorical
    le_dict = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    return df, le_dict

# ==============================
# Train model (only once, cached)
# ==============================
@st.cache_resource
def train_model():
    df, le_dict = load_data()
    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    proba = rf.predict_proba(X_test)[:, 1]

    st.sidebar.subheader("📊 Model Performance")
    st.sidebar.text(classification_report(y_test, preds))
    st.sidebar.metric("ROC-AUC", f"{roc_auc_score(y_test, proba):.3f}")

    return rf, le_dict, X.columns

# ==============================
# Main Streamlit UI
# ==============================
def main():
    st.title("💳 Loan Approval Prediction App")
    st.write("Fill in the applicant details below to predict loan approval.")

    model, le_dict, feature_names = train_model()

    # Dynamic input form
    user_data = {}
    for col in feature_names:
        if col in le_dict:  # categorical
            options = le_dict[col].classes_
            choice = st.selectbox(f"{col}", options)
            user_data[col] = le_dict[col].transform([choice])[0]
        else:  # numeric
            val = st.number_input(f"{col}", min_value=0.0, value=0.0)
            user_data[col] = val

    if st.button("🔮 Predict Loan Status"):
        input_df = pd.DataFrame([user_data])
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        status = "✅ Approved" if prediction == 1 else "❌ Rejected"
        st.subheader(f"Prediction: {status}")
        st.write(f"Approval Probability: **{prob:.2f}**")

# ==============================
if __name__ == "__main__":
    main()
