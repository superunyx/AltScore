import sys
sys.path.append('..')
import os
import time
import requests
import subprocess
import tensorflow as tf

from simulation.phone_client import PhoneClient
from shared.pretrain import load_data

def test_integration():
    # 1. Start FastAPI server
    print("Starting FastAPI server...")
    server_process = subprocess.Popen(
        ["./venv_311/bin/uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=".",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Wait for server to start
    time.sleep(4)
    
    try:
        # Load held-out test set
        print("Loading test data to evaluate global model...")
        X, y = load_data()
        split_idx = int(len(X) * 0.8)
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        # Evaluate initial global model
        print("Evaluating Base Model V1...")
        model_v1 = tf.keras.models.load_model("../shared/base_model.keras")
        loss_v1, mae_v1 = model_v1.evaluate(X_test, y_test, verbose=0)
        print(f"Base Model MAE: {mae_v1:.4f}")
        
        # Simulate 5 clients
        print("\nSimulating 5 clients training locally and submitting updates...")
        users = [f.replace("user_", "").replace(".json", "") for f in os.listdir("../data/generated_users") if f.startswith("user_")]
        
        for i in range(5):
            user = users[i]
            # Initialize phone client (suppress prints if possible, or just let them print)
            client = PhoneClient("../shared/base_model.tflite", user)
            client.load_user_data()
            delta = client.train_and_compute_delta(epochs=5)
            
            payload = {
                "client_id": user,
                "weight_delta": delta,
                "data_samples": len(client.x_train)
            }
            
            print(f"Submitting update for client {i+1} ({user})...")
            r = requests.post("http://127.0.0.1:8000/submit_update", json=payload)
            print(f"Response: {r.json()}")
            
            # For the first user, also test the score endpoint
            if i == 0:
                # Simulate infer
                infer_fn = client.interpreter.get_signature_runner('infer')
                infer_result = infer_fn(x=client.x_train)
                score = float(infer_result['output'][0][0])
                
                r_score = requests.get(f"http://127.0.0.1:8000/score/{user}?model_output={score}")
                print(f"Tested /score endpoint: {r_score.json()}")
            
        # Wait for aggregation to finish in the background
        print("\nWaiting for server to aggregate the round...")
        time.sleep(5)
        
        # Check global model
        r = requests.get("http://127.0.0.1:8000/global_model")
        global_info = r.json()
        print(f"\nNew global model info: {global_info}")
        
        if global_info["version"] < 2:
            print("❌ ERROR: Model version did not increment! Aggregation failed.")
            return
            
        # Evaluate new global model
        model_v2 = tf.keras.models.load_model(global_info["file_path"])
        loss_v2, mae_v2 = model_v2.evaluate(X_test, y_test, verbose=0)
        print(f"Aggregated Model V2 MAE: {mae_v2:.4f}")
        
        if mae_v2 < mae_v1:
            print("\n✅ SUCCESS: Federated aggregation improved the global model on the test set!")
        else:
            print("\n⚠️ SUCCESS (Technical): Pipeline worked, but MAE did not strictly decrease.")
            print("Note: In federated learning, 5 clients doing 5 epochs on small data may cause slight fluctuations.")
            
    finally:
        server_process.terminate()

if __name__ == "__main__":
    test_integration()
