import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_assets():
    model = joblib.load('best_churn_model.joblib')
    scaler = joblib.load('scaler.joblib')
    encoders = joblib.load('label_encoder.joblib')
    features = joblib.load('feature_columns.joblib')
    return model, scaler, encoders, features

model, scaler, encoders, feature_columns = load_assets()

st.set_page_config(page_title="Dashboard Prediksi Churn Lengkap - UAS", layout="wide")
st.title("🎯 Customer Churn Analytics Dashboard (Full Features)")
st.markdown("--- ")

with st.form("full_prediction_form"):
    st.subheader("📋 Input Data Pelanggan Lengkap")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Profil & Demografis**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 15, 80, 30)
        country = st.selectbox("Country", ["India", "Germany", "USA", "UK"])
        city = st.selectbox("City", ["Berlin", "Mumbai", "London", "Hamburg", "New York"])
        sub_type = st.selectbox("Subscription Type", ["Monthly", "Annual"])
        premium = st.radio("Is Premium User?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        st.write("**Aktivitas & Keterlibatan**")
        visits = st.number_input("Total Visits", 0, 100, 10)
        session_time = st.number_input("Avg Session Time (min)", 0.0, 60.0, 15.0)
        pages = st.number_input("Pages Per Session", 0.0, 20.0, 5.0)
        email_open = st.slider("Email Open Rate", 0.0, 1.0, 0.5)
        email_click = st.slider("Email Click Rate", 0.0, 1.0, 0.2)
        last_freq = st.number_input("Last 3 Month Purchase Freq", 0, 50, 5)

    with col3:
        st.write("**Finansial & Kepuasan**")
        spent = st.number_input("Total Spent ($)", 0.0, 10000.0, 500.0)
        order_val = st.number_input("Avg Order Value", 0.0, 1000.0, 50.0)
        ltv = st.number_input("Lifetime Value", 0.0, 20000.0, 1000.0)
        satisfaction = st.slider("Satisfaction Score (1-5)", 1.0, 5.0, 3.0)
        nps = st.slider("NPS Score (1-10)", 1, 10, 7)
        tickets = st.number_input("Support Tickets", 0, 50, 2)
        delay = st.number_input("Delivery Delay Days", 0, 30, 2)
        refund = st.radio("Refund Requested?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        payment = st.selectbox("Payment Method", ["UPI", "BKash", "PayPal", "Card"])

    submit = st.form_submit_button("PREDIKSI RISIKO CHURN")

if submit:
    input_data = {f: 0 for f in feature_columns}
    input_data.update({
        'gender': gender, 'age': age, 'country': country, 'city': city,
        'subscription_type': sub_type, 'is_premium_user': premium,
        'total_visits': visits, 'avg_session_time': session_time,
        'pages_per_session': pages, 'email_open_rate': email_open,
        'email_click_rate': email_click, 'last_3_month_purchase_freq': last_freq,
        'total_spent': spent, 'avg_order_value': order_val,
        'lifetime_value': ltv, 'satisfaction_score': satisfaction,
        'nps_score': nps, 'support_tickets': tickets,
        'delivery_delay_days': delay, 'refund_requested': refund,
        'payment_method': payment
    })

    df_input = pd.DataFrame([input_data])
    for col, le in encoders.items():
        if col in df_input.columns:
            try: df_input[col] = le.transform(df_input[col].astype(str))
            except: df_input[col] = -1

    X_scaled = scaler.transform(df_input[feature_columns])
    proba = model.predict_proba(X_scaled)[0][1]

    # Menurunkan threshold ke 0.40 untuk sensitivitas ekstra
    THRESHOLD = 0.40

    st.markdown("--- ")
    if proba >= THRESHOLD:
        st.error(f"### STATUS: BERISIKO CHURN (KELUAR) ⚠️")
        st.metric("Probabilitas Churn", f"{proba:.2%}")
        st.warning("Rekomendasi: Berikan penawaran khusus untuk mencegah churn.")
    else:
        st.success(f"### STATUS: RETAIN (SETIA) ✅")
        st.metric("Probabilitas Churn", f"{proba:.2%}")
        st.balloons()

# Attempting clean rebuild with sklearn 1.3.2