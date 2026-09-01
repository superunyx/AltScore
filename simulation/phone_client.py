import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta

MODEL_PATH = "../shared/base_model.tflite"
DATA_DIR = "../data/generated_users"

class PhoneClient:
    def __init__(self, tflite_model_path, user_id):
        self.user_id = user_id
        self.model_path = tflite_model_path
        
        # Load TFLite Model
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        # Extract signatures
        self.train_fn = self.interpreter.get_signature_runner('train')
        self.save_fn = self.interpreter.get_signature_runner('save')
        self.restore_fn = self.interpreter.get_signature_runner('restore')
        
        self.work_dir = f"/tmp/phone_sim_{self.user_id}"
        os.makedirs(self.work_dir, exist_ok=True)
        
        self.init_ckpt_path = os.path.join(self.work_dir, "init.ckpt")
        self.tuned_ckpt_path = os.path.join(self.work_dir, "tuned.ckpt")

    def load_user_data(self):
        file_path = os.path.join(DATA_DIR, f"user_{self.user_id}.json")
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        score = data['reliability_score']
        app_usage = {item['date']: item for item in data['app_usage']}
        sms_logs = data.get('sms_logs', [])
                
        min_date_str = min(app_usage.keys())
        min_date = datetime.strptime(min_date_str, "%Y-%m-%d")
        
        x_windows = []
        y_windows = []
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'shared')))
        from feature_engineering import compute_ratio_features

        for start_offset in range(0, 61, 5):
            window_start = min_date + timedelta(days=start_offset)
            window_end = window_start + timedelta(days=30)
            
            features = compute_ratio_features(sms_logs, app_usage, window_start, window_end)
            x_windows.append(features)
            y_windows.append([score])
            
        self.x_train = np.array(x_windows, dtype=np.float32)
        self.y_train = np.array(y_windows, dtype=np.float32)

    def extract_weights(self):
        export_fn = self.interpreter.get_signature_runner('export_weights')
        result = export_fn(dummy=tf.constant(0.0, dtype=tf.float32))
        return {k: v for k, v in result.items()}

    def train_and_compute_delta(self, epochs=5):
        # 1. Get initial model state
        init_weights = self.extract_weights()
        
        # 2. Local on-device training
        print(f"[{self.user_id}] Starting local fine-tuning for {epochs} epochs...")
        for epoch in range(epochs):
            result = self.train_fn(x=self.x_train, y=self.y_train)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"[{self.user_id}] Epoch {epoch+1}/{epochs} - Loss: {result['loss']:.6f}")
                
        # 3. Get fine-tuned model state
        tuned_weights = self.extract_weights()
        
        # 4. Compute Delta (Fine-Tuned - Base)
        delta = {}
        for key in init_weights.keys():
            diff = tuned_weights[key] - init_weights[key]
            # Convert to list for JSON serialization
            delta[key] = diff.tolist() 
            
        return delta

    def serialize_payload(self, delta):
        payload = {
            "client_id": self.user_id,
            "weight_delta": delta,
            "data_samples": len(self.x_train)
        }
        return json.dumps(payload)

def main():
    # Pick the first user from the synthetic data directory
    users = [f.replace("user_", "").replace(".json", "") for f in os.listdir(DATA_DIR) if f.startswith("user_")]
    if not users:
        print("No user data found. Please run the synthetic data generator first.")
        return
        
    target_user = users[0]
    print(f"=== Simulating phone client for user: {target_user} ===")
    
    client = PhoneClient(MODEL_PATH, target_user)
    client.load_user_data()
    
    delta = client.train_and_compute_delta(epochs=10)
    
    # Serialize for POST request
    payload_json = client.serialize_payload(delta)
    payload_bytes = payload_json.encode('utf-8')
    
    print(f"\n[{target_user}] Delta computed successfully!")
    print(f"[{target_user}] Serialized payload size: {len(payload_bytes) / 1024:.2f} KB")
    
    # Save the payload locally to inspect
    payload_file = f"payload_delta.json"
    with open(payload_file, "w") as f:
        f.write(payload_json)
    print(f"[{target_user}] Saved serialized payload to simulation/{payload_file} for inspection.")

if __name__ == "__main__":
    main()
