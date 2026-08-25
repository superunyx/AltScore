import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import subprocess
import shutil
import time

sys.path.append('..')
from shared.pretrain import load_data, build_keras_model
from simulation.phone_client import PhoneClient

def evaluate_single_device():
    X, y = load_data()
    split_idx = int(len(X) * 0.8)
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    users = [f.replace("user_", "").replace(".json", "") for f in os.listdir("../data/generated_users") if f.startswith("user_")]
    selected_users = np.random.choice(users, 5, replace=False)
    
    maes = []
    for user in selected_users:
        # Load user data
        client = PhoneClient("../shared/base_model.tflite", user)
        client.load_user_data()
        
        # Build fresh model
        model = build_keras_model()
        model.fit(client.x_train, client.y_train, epochs=25, batch_size=4, verbose=0)
        
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        maes.append(mae)
        
    return np.mean(maes)

def run_federated_seed(seed):
    # Set seed in a file or env var if needed, or just let it be random.
    # The script uses np.random.choice, so setting np.random.seed doesn't affect the subprocess unless passed.
    
    # Wipe server
    os.system("rm -rf ../server/models/ ../server/altscore.db")
    
    # Run the federated script. We'll modify it slightly to save metrics to a file so we can read it.
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    # We will let the federated script run, and we will extract the MAEs from its stdout.
    process = subprocess.Popen(
        ["../server/venv_311/bin/python", "run_federated_round.py", str(seed)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    
    maes = []
    for line in process.stdout:
        print(line, end="")
        if "Test MAE:" in line and "Global Model" in line:
            # e.g., 📊 Global Model V1 Test MAE: 0.0913 (Loss: 0.0136)
            # or Final Global Model V11 Test MAE: 0.0901
            parts = line.strip().split("MAE:")
            mae_str = parts[1].split()[0]
            maes.append(float(mae_str))
            
    process.wait()
    return maes

def main():
    print("--- 1. Evaluating Single-Device-Only Baseline ---")
    single_mae = evaluate_single_device()
    print(f"Average Single-Device MAE: {single_mae:.4f}")
    
    print("\n--- 2. Central Pretrained Baseline ---")
    X, y = load_data()
    split_idx = int(len(X) * 0.8)
    X_test, y_test = X[split_idx:], y[split_idx:]
    base_model = tf.keras.models.load_model("../shared/base_model.keras")
    _, central_mae = base_model.evaluate(X_test, y_test, verbose=0)
    print(f"Central Base Model V1 MAE: {central_mae:.4f}")
    
    print("\n--- 3. Running Multi-Seed Federated Learning ---")
    all_maes = []
    for seed in [42, 100, 999]:
        print(f"\n>>> Running Federated Simulation (Seed {seed})")
        maes = run_federated_seed(seed)
        # We expect 11 values (V1 up to V11)
        if len(maes) >= 11:
            all_maes.append(maes[:11])
        else:
            print("Run failed or didn't complete 10 rounds.")
        time.sleep(3)
            
    all_maes = np.array(all_maes)
    mean_maes = np.mean(all_maes, axis=0)
    min_maes = np.min(all_maes, axis=0)
    max_maes = np.max(all_maes, axis=0)
    
    print("\n=== FINAL RESULTS ===")
    print(f"Single Device: {single_mae:.4f}")
    print(f"Central Pretrained: {central_mae:.4f}")
    for i, m in enumerate(mean_maes):
        print(f"Federated V{i+1}: Mean {m:.4f} (Min: {min_maes[i]:.4f}, Max: {max_maes[i]:.4f})")
        
    # Plotting
    plt.figure(figsize=(9, 6))
    rounds = np.arange(1, 12)
    plt.plot(rounds, mean_maes, marker='o', linestyle='-', color='b', linewidth=2, label='Federated Mean MAE')
    plt.fill_between(rounds, min_maes, max_maes, color='b', alpha=0.2, label='Min/Max Range')
    
    plt.axhline(y=single_mae, color='r', linestyle='--', label='Single Device Only')
    plt.axhline(y=central_mae, color='g', linestyle='--', label='Central Pretrained (V1)')
    
    plt.title('Federated Learning Progress vs Baselines', fontsize=14)
    plt.xlabel('Global Model Version (Round + 1)', fontsize=12)
    plt.ylabel('Mean Absolute Error (Lower is Better)', fontsize=12)
    plt.xticks(rounds)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path = "federated_learning_progress.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nChart saved to simulation/{plot_path}")

if __name__ == "__main__":
    main()
