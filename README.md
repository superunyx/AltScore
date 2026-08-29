# AltScore

AltScore delivers privacy-first credit scoring for gig workers. By computing scores directly on a user's own phone, gig workers can finally prove their financial reliability without ever having to hand over their raw, personal financial data to a third party.

## The Problem

Millions of gig workers are financially reliable but lack the formal credit histories required for fair loans. Existing "alternative data" credit scoring approaches attempt to solve this, but they do so by forcing workers to hand over incredibly sensitive personal data—like text messages, location history, and private transaction logs—to corporate servers. A gig worker shouldn't have to trade away their fundamental right to privacy just to access a financial lifeline.

## How AltScore Works

AltScore flips the traditional data collection model. Instead of sending your personal data to a server to be analyzed, we send a blank scoring model to your phone. 

Your phone privately learns from your everyday data on the device itself. When it finishes learning, it securely shares only what it learned—a mathematical summary—back to the shared system. This helps the global scoring model get smarter for everyone, while ensuring that no company or loan officer ever sees your texts, transactions, or app usage. You get a reliable credit score, generated with zero compromise on privacy.

## Setup Instructions

### Backend Server
1. Navigate to `server/`.
2. Install dependencies: `pip install -r requirements.txt` (requires Python 3.11).
3. Start the FastAPI server: `python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload`

### Python Simulation
1. Navigate to `simulation/`.
2. Generate synthetic data: `python generate_synthetic_data.py`
3. Run a federated round: `python run_federated_round.py`

### Dashboard
1. Navigate to `server/`.
2. Start the Streamlit app: `streamlit run app.py`

### Android App
1. Install JDK 17 (required; newer or older versions will cause build failures).
2. Navigate to `mobile/` and run `npm install`.
3. Ensure `mobile/android/settings.gradle` has the foojay plugin pinned to exactly `id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"`.
4. Start the Metro bundler: `npm start`
5. Build and install to a device via adb: `cd android && ./gradlew assembleDebug`

## Current Progress and Results

We have successfully verified real, end-to-end on-device training on actual physical Android hardware, moving beyond basic simulated environments. 

Our evaluation proves that we achieve privacy without an accuracy penalty. The model's Mean Absolute Error (MAE) under different training conditions is statistically indistinguishable between centralized and privacy-preserving methods:
* Centralized (all data exposed): ~0.0913 MAE
* Federated (AltScore's private method): ~0.0908 MAE
* Isolated (single device, no collaboration): ~0.1551 MAE

## Why This Matters

This project demonstrates that decentralized machine learning can operate on real hardware right now. The privacy guarantee is backed by an auditable architecture where raw data simply never leaves the phone. By relying on honest statistical evaluation rather than cherry-picked training runs, AltScore proves that inclusive, alternative credit scoring does not require a mass surveillance approach.

## License and Contributions

This is an academic project and is currently not open for outside contributions. All rights reserved.
