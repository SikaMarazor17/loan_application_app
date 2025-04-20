# ============ ALL IMPORTS FIRST ============
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import sqlite3
import os
from datetime import datetime
import time

# ============ NON-STREAMLIT FUNCTION DEFINITIONS ============
def init_db_connection():
    os.makedirs("data", exist_ok=True)
    db_path = os.path.join("data", "bank_risk_100k.db")
    return sqlite3.connect(db_path, check_same_thread=False)

def generate_data(n=100000):
    np.random.seed(42)
    data = pd.DataFrame({
        'income': np.random.randint(120000, 1500000, n),
        'credit_score': np.random.randint(300, 850, n),
        'loan_amount': np.random.randint(50000, 5000000, n),
        'debt_to_income': np.round(np.random.uniform(0.1, 0.5, n), 2)
    })
    data['default_risk'] = np.where(
        (data['loan_amount'] > 0.5 * data['income']) |
        (data['credit_score'] < 550) |
        (data['debt_to_income'] > 0.35),
        1, 0
    )
    return data

def train_model(data):
    X = data[['income', 'credit_score', 'loan_amount', 'debt_to_income']]
    y = data['default_risk']
    model = LogisticRegression()
    model.fit(X, y)
    return model

def get_loan_decision(model, income, credit_score, loan_amount, dti):
    input_data = pd.DataFrame([[income, credit_score, loan_amount, dti]],
                            columns=['income', 'credit_score', 'loan_amount', 'debt_to_income'])
    default_prob = model.predict_proba(input_data)[0][1]
    
    if dti > 0.4:
        return "❌ Rejected: Debt-to-income ratio exceeds 40%", None, default_prob
    elif credit_score < 550:
        return "❌ Rejected: Credit score below 550 (NCA guidelines)", None, default_prob
    elif default_prob > 0.65:
        return "❌ Rejected: High default risk for SA market", None, default_prob
    elif default_prob > 0.35:
        counter_offer = loan_amount * 0.7
        return f"⚠️ Approved with adjustment: Offered: R{counter_offer:,.0f}", counter_offer, default_prob
    else:
        return "✅ Approved", loan_amount, default_prob

# ============ STREAMLIT IMPORT & CONFIG (FIRST ST COMMAND) ============
import streamlit as st
st.set_page_config(page_title="SA Loan Approval System", layout="centered")

# ============ STREAMLIT CACHED RESOURCES ============
@st.cache_resource
def load_model():
    data = generate_data()
    return train_model(data)

@st.cache_resource
def get_db_connection():
    return init_db_connection()

model = load_model()
conn = get_db_connection()

# ============ STREAMLIT UI COMPONENTS ============
st.title("🏦 Nedbank Loan Approval")
st.markdown("This app evaluates loans based on **South African credit regulations** and **Nedbank specific internal risk management rules**.")

# Input Section
col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Annual Income (R)", min_value=120000, max_value=1500000, value=150000, step=10000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)
    
with col2:
    loan_amount = st.number_input("Loan Amount (R)", min_value=50000, max_value=5000000, value=50000, step=10000)
    dti = st.number_input("Debt-to-Income Ratio", min_value=0.1, max_value=0.8, value=0.3, step=0.01, format="%.2f")

# Decision Logic
if st.button("Check Approval", type="primary", key="check_approval_button"):
    decision, offer, prob = get_loan_decision(model, income, credit_score, loan_amount, dti)
    
    # Display results
    st.subheader("Result:")
    st.write(decision)
    if decision.startswith("⚠️"):
        st.metric("Adjusted Loan Amount", f"R{offer:,.0f}")
        st.caption(f"Originally requested: R{loan_amount:,.0f}")
    elif decision.startswith("✅"):
        st.metric("Approved Amount", f"R{loan_amount:,.0f}")
    st.info(f"Predicted default probability: {prob:.0%}")
    
    # Database logging
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        branch = np.random.choice(["Cape Town", "Sandton", "Durban"])
        officer_prefix = {"Cape Town": "CPT", "Sandton": "JHB", "Durban": "DBN"}[branch]
        officer_id = f"MGR_{officer_prefix}_{np.random.randint(1, 21):03d}"
        adjusted_amt = offer if decision.startswith("⚠️") else None
        reason = decision.split(":")[-1].strip() if ":" in decision else ""
        
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO loan_decisions (
            timestamp, branch, income, credit_score, loan_amount,
            adjusted_amount, dti, decision, reason, default_prob, officer_id
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ''', (
            timestamp, branch, income, credit_score, loan_amount,
            adjusted_amt, dti, 
            "APPROVED" if decision.startswith("✅") else "ADJUSTED" if decision.startswith("⚠️") else "REJECTED",
            reason, prob, officer_id
        ))
        conn.commit()
    except Exception as e:
        st.error(f"Database error: {str(e)}")

# Help Section
with st.expander("ℹ️ How decisions are made", expanded=False):
    st.markdown("""
    ### South African Lending Rules:
    - **Credit Score:**
      - <550: Automatic rejection (NCA)
      - 550-600: High scrutiny
      - 600+: Preferred

    - **Debt-to-Income (DTI):**
      - >40%: Automatic rejection

    - **Model Predictions:**
      - >65% default risk: Rejected
      - 35-65% risk: Counteroffer (70%)
      - <35% risk: Approved
    """, unsafe_allow_html=True)
