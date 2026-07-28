import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Explainable Loan Risk Intelligence System",
                   page_icon="🏦",
                   layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("models/loan_model.pkl")

model = load_model()

st.title("🏦 Explainable Loan Risk Intelligence System")
st.caption("Predict loan approval using a tuned Random Forest classifier.")

st.sidebar.header("Project Information")
st.sidebar.success("Model: Random Forest")
st.sidebar.metric("Test Accuracy", "97.07%")
st.sidebar.markdown("""
**Top Features**
- CIBIL Score
- Loan Amount
- Loan Term
- Income
""")

st.markdown("### Applicant Details")

col1, col2 = st.columns(2)

with col1:
    dependents = st.number_input("Number of Dependents", 0, 20, 1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    income = st.number_input("Annual Income (₹)", min_value=0, value=8000000)
    loan_term = st.number_input("Loan Term (Months)", min_value=1, value=12)
    residential = st.number_input("Residential Assets Value (₹)", min_value=0, value=6000000)
    luxury = st.number_input("Luxury Assets Value (₹)", min_value=0, value=1500000)

with col2:
    self_emp = st.selectbox("Self Employed", ["No", "Yes"])
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=2000000)
    cibil = st.number_input("CIBIL Score", min_value=300, max_value=900, value=800)
    commercial = st.number_input("Commercial Assets Value (₹)", min_value=0, value=2000000)
    bank = st.number_input("Bank Assets Value (₹)", min_value=0, value=1000000)

edu = 1 if education == "Graduate" else 0
emp = 1 if self_emp == "Yes" else 0

features = np.array([[
    dependents,
    edu,
    emp,
    income,
    loan_amount,
    loan_term,
    cibil,
    residential,
    commercial,
    luxury,
    bank
]])

if st.button("🔍 Predict Loan Status", use_container_width=True):
    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]

    st.markdown("## Prediction Result")

    if pred == 1:
        st.success("✅ Loan Approved")
        st.metric("Approval Confidence", f"{probs[1]*100:.2f}%")
        st.info("""**Possible Reasons**
- Strong CIBIL score
- Healthy financial profile
- Suitable loan amount
- Good asset value""")
    else:
        st.error("❌ Loan Rejected")
        st.metric("Rejection Confidence", f"{probs[0]*100:.2f}%")
        st.warning("""**Possible Reasons**
- Low CIBIL score
- High loan burden
- Limited financial assets""")

    st.markdown("### Applicant Summary")
    st.write({
        "Dependents": dependents,
        "Education": education,
        "Self Employed": self_emp,
        "Annual Income": income,
        "Loan Amount": loan_amount,
        "Loan Term": loan_term,
        "CIBIL": cibil
    })

with st.expander("📘 About the Model"):
    st.markdown("""
This application predicts loan approval using a tuned **Random Forest Classifier**.

### Workflow
1. Data preprocessing
2. Feature engineering
3. Random Forest training
4. SHAP explainability
5. Streamlit deployment

### Technologies
- Python
- Scikit-learn
- Streamlit
- SHAP
- Pandas
- NumPy
""")

st.markdown("---")
st.caption("Developed as part of an Explainable AI based Loan Risk Intelligence System.")
