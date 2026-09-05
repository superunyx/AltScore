import sys
import os
import time
import requests
import subprocess
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append('..')
from simulation.phone_client import PhoneClient
from shared.pretrain import load_data, OnDeviceModel
from shared.differential_privacy import clip_and_add_noise
from shared.crypto_utils import encrypt_payload

SERVER_URL = os.environ.get("ALTSCORE_SERVER_URL", "http://127.0.0.1:8000")
NUM_ROUNDS = 10
CLIENTS_PER_ROUND = 5
DATA_DIR = "../data/generated_users"

def convert_to_tflite(keras_model_path, output_tflite_path):
    print(f"⚙️  Converting {keras_model_path} to TFLite for devices...")
    model = tf.keras.models.load_model(keras_model_path)
    on_device_model = OnDeviceModel(model)
    
    # Save via tf.saved_model to preserve signatures
    saved_model_dir = "temp_saved_model"
    tf.saved_model.save(
        on_device_model, 
        saved_model_dir,
        signatures={
            'train': on_device_model.train.get_concrete_function(),
            'infer': on_device_model.infer.get_concrete_function(),
            'save': on_device_model.save.get_concrete_function(),
            'restore': on_device_model.restore.get_concrete_function(),
            'export_weights': on_device_model.export_weights.get_concrete_function(),
        }
    )
    
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()
    
    with open(output_tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"⚙️  Saved TFLite model to {output_tflite_path}")

def main():
    if len(sys.argv) > 1:
        np.random.seed(int(sys.argv[1]))
    
    print("Starting Live FastAPI Server...")
    server_process = subprocess.Popen(
        ["../server/venv_311/bin/uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd="../server",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(5)
    
    try:
        # Load test set
        X, y = load_data()
        split_idx = int(len(X) * 0.8)
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        users = [f.replace("user_", "").replace(".json", "") for f in os.listdir(DATA_DIR) if f.startswith("user_")]
        
        # Track metrics
        round_history = []
        mae_history = []
        
        for round_num in range(1, NUM_ROUNDS + 1):
            print(f"\n{'='*50}")
            print(f"🚀 STARTING FEDERATED ROUND {round_num}/{NUM_ROUNDS}")
            print(f"{'='*50}")
            
            # Get latest global model
            
            # Get latest global model
            r = requests.get(f"{SERVER_URL}/global_model")
            global_info = r.json()
            model_version = global_info["version"]
            
            # Get public key
            r_pub = requests.get(f"{SERVER_URL}/public_key")
            server_pub_pem = r_pub.json()["public_key_pem"]

            
            keras_path = f"../server/{global_info['file_path']}"
            tflite_path = f"global_model_v{model_version}.tflite"
            
            # Evaluate Current Global Model
            model = tf.keras.models.load_model(keras_path)
            loss, mae = model.evaluate(X_test, y_test, verbose=0)
            print(f"📊 Global Model V{model_version} Test MAE: {mae:.4f} (Loss: {loss:.4f})")
            
            round_history.append(round_num)
            mae_history.append(mae)
            
            # Convert for clients
            convert_to_tflite(keras_path, tflite_path)
            
            # Select random users for this round
            selected_users = np.random.choice(users, CLIENTS_PER_ROUND, replace=False)
            
            for user in selected_users:
                print(f"\n📱 Simulating device for {user[:8]}...")
                print(f"   ↳ Loading starting model: {tflite_path}")
                client = PhoneClient(tflite_path, user)
                client.load_user_data()
                delta = client.train_and_compute_delta(epochs=10)
                
                # Calculate L2 norm of the delta
                l2_norm = np.sqrt(sum(np.sum(np.square(np.array(d))) for d in delta.values()))
                print(f"   ↳ Weight Delta L2 Norm: {l2_norm:.6f}")
                
                
                noised_delta, dp_stats = clip_and_add_noise(delta)
                print(f"   ↳ DP Stats: original_l2={dp_stats['original_l2_norm']:.4f}, clip_factor={dp_stats['clip_factor_applied']:.4f}, post_noise_l2={dp_stats['post_noise_l2_norm']:.4f}")
                
                inner_payload = {
                    "weight_delta": noised_delta,
                    "data_samples": len(client.x_train)
                }
                
                encrypted_envelope = encrypt_payload(inner_payload, server_pub_pem)
                
                payload = {
                    "client_id": user,
                    "encrypted_key": encrypted_envelope["encrypted_key"],
                    "nonce": encrypted_envelope["nonce"],
                    "ciphertext": encrypted_envelope["ciphertext"]
                }

                
                r = requests.post(f"{SERVER_URL}/submit_update", json=payload)
                print(f"✅ Update sent. Server: {r.json()['message']}")
                
            # Wait for aggregation
            print(f"\n⏳ Waiting for server to aggregate round {round_num}...")
            for _ in range(30):
                time.sleep(1)
                r = requests.get(f"{SERVER_URL}/global_model")
                if r.json()["version"] > model_version:
                    print(f"✅ Server aggregated new global model V{r.json()['version']}")
                    break
            else:
                print("❌ Server failed to aggregate in time!")
                break
                
        # Final evaluation
        r = requests.get(f"{SERVER_URL}/global_model")
        final_info = r.json()
        keras_path = f"../server/{final_info['file_path']}"
        model = tf.keras.models.load_model(keras_path)
        final_loss, final_mae = model.evaluate(X_test, y_test, verbose=0)
        
        round_history.append(NUM_ROUNDS + 1)
        mae_history.append(final_mae)
        
        print(f"\n🎉 FEDERATED LEARNING COMPLETE")
        print(f"Final Global Model V{final_info['version']} Test MAE: {final_mae:.4f}")
        
        # Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(round_history, mae_history, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
        plt.title('Federated Learning Progress: Global Model Test MAE', fontsize=14)
        plt.xlabel('Evaluation Checkpoint (Round)', fontsize=12)
        plt.ylabel('Mean Absolute Error (Lower is Better)', fontsize=12)
        plt.xticks(round_history)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = "federated_learning_progress.png"
        plt.savefig(plot_path, dpi=300)
        print(f"\n📈 Chart saved to simulation/{plot_path}")
        
    finally:
        server_process.terminate()

if __name__ == "__main__":
    main()
