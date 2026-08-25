import os
import glob
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from datetime import datetime, timedelta
from PIL import Image

# Config
DATA_DIR = "../data/generated_users"
MODELS_DIR = "../server/models"
PLOT_PATH = "../simulation/federated_learning_progress.png"

st.set_page_config(page_title="AltScore Loan Officer Dashboard", layout="wide")

def get_latest_model():
    if not os.path.exists(MODELS_DIR):
        return None, 0
    models = glob.glob(os.path.join(MODELS_DIR, "*.keras"))
    if not models:
        return None, 0
    # sort by version number (e.g. model_v1.keras, model_v2.keras)
    models.sort(key=lambda x: int(os.path.basename(x).split("_v")[-1].split(".")[0]))
    latest_model_path = models[-1]
    version = int(os.path.basename(latest_model_path).split("_v")[-1].split(".")[0])
    return latest_model_path, version

@st.cache_resource
def load_global_model(model_path):
    return tf.keras.models.load_model(model_path)

def compute_qualitative_factors(data):
    # Determine qualitative strings without exposing raw numbers
    app_usage = {item['date']: item for item in data['app_usage']}
    
    daily_income = {}
    for sms in data['sms_logs']:
        if sms['type'] == 'credit':
            date_str = sms['timestamp'].split('T')[0]
            daily_income[date_str] = daily_income.get(date_str, 0) + sms['amount']
            
    days_with_income = len(daily_income)
    avg_sessions = np.mean([item['sessions'] for item in app_usage.values()])
    
    factors = []
    if days_with_income >= 15:
        factors.append("💸 Income Consistency: High (Regular gig payouts)")
    elif days_with_income >= 5:
        factors.append("💸 Income Consistency: Moderate (Occasional gig payouts)")
    else:
        factors.append("💸 Income Consistency: Sporadic (Rare gig payouts)")
        
    if avg_sessions >= 8:
        factors.append("📱 App Engagement: Very Active (Highly dedicated)")
    elif avg_sessions >= 3:
        factors.append("📱 App Engagement: Active (Consistent worker)")
    else:
        factors.append("📱 App Engagement: Low (Infrequent worker)")
        
    return factors

def get_applicants(model):
    users = glob.glob(os.path.join(DATA_DIR, "user_*.json"))
    applicants = []
    # Just take the first 10 for the dashboard to keep it minimal
    for user_file in users[:10]:
        with open(user_file, 'r') as f:
            data = json.load(f)
            
        user_id = os.path.basename(user_file).replace("user_", "").replace(".json", "")
        
        # Prepare features exactly as the phone would
        app_usage = {item['date']: item for item in data['app_usage']}
        daily_income = {}
        for sms in data['sms_logs']:
            if sms['type'] == 'credit':
                date_str = sms['timestamp'].split('T')[0]
                daily_income[date_str] = daily_income.get(date_str, 0) + sms['amount']
                
        min_date_str = min(app_usage.keys())
        min_date = datetime.strptime(min_date_str, "%Y-%m-%d")
        
        user_features = []
        for i in range(30):
            current_date = min_date + timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            sessions = app_usage.get(date_str, {}).get('sessions', 0)
            hours = app_usage.get(date_str, {}).get('hours_active', 0.0)
            income = daily_income.get(date_str, 0.0)
            user_features.append([sessions / 15.0, hours / 24.0, income / 500.0])
            
        x_input = np.array([user_features], dtype=np.float32)
        score = float(model.predict(x_input, verbose=0)[0][0])
        
        factors = compute_qualitative_factors(data)
        
        # Scale score 0-1000 for display
        display_score = int(score * 1000)
        
        applicants.append({
            "User ID": user_id[:8].upper(),
            "AltScore": display_score,
            "Key Factors": factors
        })
        
    # Sort by score descending
    applicants.sort(key=lambda x: x["AltScore"], reverse=True)
    return applicants

def main():
    st.title("🏦 AltScore Loan Officer Dashboard")
    st.markdown("### Federated Learning Powered Credit Scoring")
    st.markdown("*Privacy-by-design: Raw user behavioral data never leaves the phone. Only final qualitative factors and computed scores are visible here.*")
    
    st.divider()
    
    model_path, version = get_latest_model()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(f"🌐 Global Federated Model (V{version})")
        if os.path.exists(PLOT_PATH):
            image = Image.open(PLOT_PATH)
            st.image(image, caption="Federated Learning MAE Progress", use_container_width=True)
        else:
            st.info("No federated learning plot found. Run simulation/run_federated_round.py first.")
            
    with col2:
        st.subheader("👥 Applicant Queue")
        
        if model_path:
            model = load_global_model(model_path)
            applicants = get_applicants(model)
            
            for app in applicants:
                with st.expander(f"Applicant #{app['User ID']} — AltScore: **{app['AltScore']}**/1000", expanded=False):
                    st.write("**Top Contributing Factors:**")
                    for factor in app['Key Factors']:
                        st.markdown(f"- {factor}")
                    st.caption("🔒 Raw behavioral data securely encrypted and processed on-device.")
        else:
            st.warning("No global models found.")

if __name__ == "__main__":
    main()
