import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import sqlite3
import time

# Simulate data with South African context
def generate_data(n=100000):
    np.random.seed(42)
    data = pd.DataFrame({
        'income': np.random.randint(120000, 1500000, n),  # R120k to R1.5M
        'credit_score': np.random.randint(300, 850, n),
        'loan_amount': np.random.randint(50000, 5000000, n),  # R50k to R5M
        'debt_to_income': np.round(np.random.uniform(0.1, 0.5, n), 2)
    })
    # Realistic default rules (aligned with SA banking)
    data['default_risk'] = np.where(
        (data['loan_amount'] > 0.5 * data['income']) |  # Large loan relative to income
        (data['credit_score'] < 550) |                  # Subprime credit
        (data['debt_to_income'] > 0.35),               # High debt burden
        1, 0
    )
    return data

# Train the model
def train_model(data):
    X = data[['income', 'credit_score', 'loan_amount', 'debt_to_income']]
    y = data['default_risk']
    model = LogisticRegression()
    model.fit(X, y)
    return model

# Decision logic for SA context
def get_loan_decision(model, income, credit_score, loan_amount, dti):
    input_data = pd.DataFrame([[income, credit_score, loan_amount, dti]],
                            columns=['income', 'credit_score', 'loan_amount', 'debt_to_income'])
    
    default_prob = model.predict_proba(input_data)[0][1]  # P(default)
    
    # SA-specific rules
    if dti > 0.4:
        return "❌ Rejected: DTI exceeds 40%", None, default_prob
    elif credit_score < 550:
        return "❌ Rejected: Credit score <550", None, default_prob
    elif default_prob > 0.65:
        return "❌ Rejected: High default risk", None, default_prob
    elif default_prob > 0.35:
        counter_offer = loan_amount * 0.7  # Reduce by 30%
        return f"⚠️ Approved with adjustment: Offered: R{counter_offer:,.0f}", counter_offer, default_prob
    else:
        return "✅ Approved", loan_amount, default_prob

# Streamlit UI
st.set_page_config(page_title="SA Loan Approval System", layout="centered")
st.title("🏦 South African Loan Approval")
st.markdown("This app evaluates loans based on **South African credit regulations** and **bank specific internal risk management rules**.")

# Generate data and train model
data = generate_data()
model = train_model(data)

# Input Section - Only number inputs
col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Annual Income (R)", min_value=120000, max_value=1500000, value=150000, step=10000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)
    
with col2:
    loan_amount = st.number_input("Loan Amount (R)", min_value=50000, max_value=5000000, value=50000, step=10000)
    dti = st.number_input("Debt-to-Income Ratio", min_value=0.1, max_value=0.8, value=0.3, step=0.01, format="%.2f")

# Single button that handles both decision display and database logging
if st.button("Check Approval", type="primary", key="check_approval_button"):
    # Get decision first
    decision, offer, prob = get_loan_decision(model, income, credit_score, loan_amount, dti)
    
    # Show result to user
    st.subheader("Result:")
    st.write(decision)
    if decision.startswith("⚠️"):  # Adjusted offer
        st.metric("Adjusted Loan Amount", f"R{offer:,.0f}")
        st.caption(f"Originally requested: R{loan_amount:,.0f}")
    elif decision.startswith("✅"):  # Full approval
        st.metric("Approved Amount", f"R{loan_amount:,.0f}")
    st.info(f"Predicted default probability: {prob:.0%}")
    
    # Log to database
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    branch = np.random.choice(["Cape Town", "Sandton", "Durban"])
    officer_prefix = {
        "Cape Town": "CPT",
        "Sandton": "JHB", 
        "Durban": "DBN"
    }[branch]
    officer_id = f"MGR_{officer_prefix}_{np.random.randint(1, 21):03d}"
    
    adjusted_amt = offer if decision.startswith("⚠️") else None
    reason = decision.split(":")[-1].strip() if ":" in decision else ""
    
    try:
        conn = sqlite3.connect('bank_risk_100k.db')
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO loan_decisions (
            timestamp, branch, income, credit_score, loan_amount,
            adjusted_amount, dti, decision, reason, default_prob, officer_id
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ''', (
            timestamp, branch, income, credit_score, loan_amount,
            adjusted_amt, dti, 
            "APPROVED" if decision.startswith("✅") else "ADJUSTED" if decision.startswith("⚠️") else "REJECTED_RISK" if decision.startswith("High default risk") else "REJECTED_DTI" if decision.startswith("Rejected: DTI exceeds 40%") else "REJECTED_CREDIT"
            "Low risk profile" if decision.startswith("✅") else reason, prob, officer_id
        ))
        conn.commit()
        #st.success("Decision logged to database!")
        #time.sleep(0.5)  # Brief pause before balloons
        #st.balloons()
    except Exception as e:
        st.error(f"Database error: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

# SA-specific explanations
with st.expander("ℹ️ How decisions are made", expanded=False):
    st.markdown("""
### South African Lending Rules:

- **Credit Score:**
  - <span style='color:red'>&lt;550</span>: Automatic rejection (per National Credit Act)
  - <span style='color:orange'>550-600</span>: High scrutiny
  - <span style='color:green'>600+</span>: Preferred

- **Debt-to-Income (DTI):**
  - <span style='color:red'>&gt;40%</span>: Automatic rejection

- **Model Predictions:**
  - <span style='color:red'>&gt;65%</span> default risk: Rejected
  - <span style='color:orange'>35-65%</span> risk: Counteroffer (70% of requested amount)
  - <span style='color:green'>&lt;35%</span> risk: Approved
""", unsafe_allow_html=True)