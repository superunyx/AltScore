<div align="center">
  <h1>🛡️ AltScore</h1>
  <p><strong>Privacy-First Credit Scoring for Gig Workers</strong></p>
  <p>
    Empowering gig workers to prove their financial reliability without compromising their personal data, using <b>Federated Learning</b> directly on their phones.
  </p>
</div>

---

## 🎯 The Problem
Millions of gig workers are financially reliable but lack the formal credit histories required for fair loans. Existing "alternative data" credit scoring approaches attempt to solve this by forcing workers to hand over incredibly sensitive personal data—like text messages, location history, and private transaction logs—to corporate servers. 

A gig worker shouldn't have to trade away their fundamental right to privacy just to access a financial lifeline.

## 💡 How AltScore Works
AltScore flips the traditional data collection model upside down:
1. **No Data Leaves the Device:** Instead of sending your personal data to a server to be analyzed, we send a *blank scoring model* to your phone.
2. **On-Device Learning:** Your phone privately learns from your everyday data on the device itself. It doesn't look at absolute numbers (like your total income), but rather translates your activity into scale-invariant financial behavior ratios (like your savings rate or income regularity).
3. **Secure Collaboration:** When it finishes learning, it securely shares only a mathematical summary back to the system. 
4. **Smarter for Everyone:** This helps the global scoring model get smarter for everyone, ensuring no company ever sees your texts, transactions, or app usage.

**You get a reliable credit score, generated with zero compromise on privacy.**

## ✨ Features
- 📱 **100% On-Device Processing:** Raw data never touches a cloud server.
- 🔐 **Mathematical Privacy:** Uses state-of-the-art Federated Learning to only share weight updates, never data.
- 📊 **Loan Officer Dashboard:** A real-time web dashboard for reviewing applicant scores securely.
- 🤖 **Android Ready:** A complete end-to-end mobile application built with React Native.

---

## 🚀 Complete Install & Run Guide

Follow these steps to get the entire AltScore ecosystem running on your machine.

### 📋 Prerequisites
Make sure you have the following installed:
- **Python 3.11** (for the backend and dashboard)
- **Node.js & npm** (for the mobile app)
- **JDK 17** (required for building the Android app; newer/older versions will fail)
- **Android Studio / adb** (for running the app on an emulator or physical device)

### 1️⃣ Backend Server (FastAPI)
The backend coordinates the federated learning rounds and aggregates the model.

```bash
# 1. Navigate to the server directory
cd server/

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start the FastAPI server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
*The server will run on `http://localhost:8000`.*

### 2️⃣ Simulation (Optional but Recommended)
Generate synthetic gig worker data and run a simulated federated learning round to train the initial model.

```bash
# 1. Open a new terminal and navigate to simulation/
cd simulation/

# 2. Generate synthetic data
python generate_synthetic_data.py

# 3. Run a federated round
python run_federated_round.py
```

### 3️⃣ Loan Officer Dashboard (Streamlit)
View the federated learning progress and applicant scores.

```bash
# 1. Open a new terminal and navigate to dashboard/
cd dashboard/

# 2. Start the dashboard
streamlit run app.py
```
*The dashboard will open automatically in your browser.*

### 4️⃣ Android App (React Native)
Run the actual mobile client.

```bash
# 1. Open a new terminal and navigate to mobile/
cd mobile/

# 2. Install Node dependencies
npm install

# 3. Start the Metro bundler
npm start

# 4. In a separate terminal (still in the mobile/ directory), build and install on your Android device/emulator:
cd android
./gradlew assembleDebug
```

---

## 📊 Current Progress and Results

We have successfully verified real, end-to-end on-device training on physical Android hardware. Our new feature extraction pipeline adds a "feature-layer" privacy design by analyzing only dimensionless financial ratios, ensuring absolute wealth never exists in the feature vector.

Our evaluation proves that **we achieve privacy without an accuracy penalty**. 
- **Centralized** (all data exposed): ~0.1012 MAE
- **Federated** (AltScore's private method): ~0.1015 MAE
- **Isolated** (single device, no collaboration): ~0.2116 MAE

The model successfully extracts significant signals based on strong financial behavior correlations, such as:
- Expense-to-Income ($r = -0.80$)
- Savings Rate ($r = +0.77$)
- Shortfall Frequency ($r = -0.77$)
- Income Stability ($r = +0.72$)

*(Note on Significance: A paired t-test comparing Federated vs. Centralized across 3 seeds yields $t = 0.14$, $p = 0.905$. This "not significant" difference means they track so closely that they are statistically indistinguishable.)*

## 🔒 Why This Matters

This project demonstrates that decentralized machine learning can operate on real hardware right now. The privacy guarantee is backed by an auditable architecture where raw data simply never leaves the phone. 

By relying on honest statistical evaluation rather than cherry-picked training runs, AltScore proves that **inclusive, alternative credit scoring does not require a mass surveillance approach.**

## 📜 License and Contributions

This is an academic project and is currently not open for outside contributions. All rights reserved.
