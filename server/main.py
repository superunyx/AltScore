import os
import json
import sqlite3
import numpy as np
import tensorflow as tf

import sys
sys.path.append('../shared')
from crypto_utils import generate_server_keypair, load_public_key_pem, load_private_key, decrypt_payload
from fastapi import HTTPException
from fastapi import FastAPI, BackgroundTasks, Query
from pydantic import BaseModel
from typing import Dict, List, Any

app = FastAPI(title="AltScore Federated Learning Server")

MIN_CLIENTS_PER_ROUND = 5
DB_PATH = "altscore.db"
MODEL_DIR = "models"
BASE_MODEL_PATH = "../shared/base_model.keras"

class UpdatePayload(BaseModel):
    client_id: str
    encrypted_key: str
    nonce: str
    ciphertext: str

# In-memory queue for current round
current_round_updates: List[UpdatePayload] = []
round_lock = False

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS clients (client_id TEXT PRIMARY KEY, registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS rounds (round_id INTEGER PRIMARY KEY AUTOINCREMENT, num_clients INTEGER, aggregated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS global_models (version INTEGER PRIMARY KEY AUTOINCREMENT, file_path TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def setup():
    generate_server_keypair('keys')
    global private_key
    private_key = load_private_key('keys')

    os.makedirs(MODEL_DIR, exist_ok=True)
    init_db()
    # Check if we have at least version 1
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT version FROM global_models ORDER BY version DESC LIMIT 1")
    row = c.fetchone()
    if not row:
        # Save V1
        v1_path = os.path.join(MODEL_DIR, "model_v1.keras")
        if os.path.exists(BASE_MODEL_PATH):
            model = tf.keras.models.load_model(BASE_MODEL_PATH)
            model.save(v1_path)
            c.execute("INSERT INTO global_models (version, file_path) VALUES (1, ?)", (v1_path,))
            conn.commit()
    conn.close()

@app.on_event("startup")
def startup():
    setup()

def get_current_model():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT version, file_path FROM global_models ORDER BY version DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row:
        return row[0], row[1]
    return None, None

def aggregate_updates():
    global current_round_updates, round_lock
    if len(current_round_updates) < MIN_CLIENTS_PER_ROUND:
        return
        
    round_lock = True
    print(f"Starting aggregation for {len(current_round_updates)} clients.")
    
    # Grab updates and clear queue
    updates = current_round_updates[:MIN_CLIENTS_PER_ROUND]
    current_round_updates = current_round_updates[MIN_CLIENTS_PER_ROUND:]
    
    version, model_path = get_current_model()
    model = tf.keras.models.load_model(model_path)
    base_weights = {w.name: w.numpy() for w in model.weights}
    
    # FedAvg implementation
    results = []
    total_samples = sum(u.data_samples for u in updates)
    
    for update in updates:
        delta = update.weight_delta
        samples = update.data_samples
        
        # Reconstruct fine-tuned weights for this client
        client_weights = []
        for w in model.weights:
            name = w.name
            d = np.array(delta[name], dtype=np.float32)
            # Tuned = Base + Delta
            client_weights.append(base_weights[name] + d)
            
        results.append((client_weights, samples))
        
    # Average the weights (FedAvg)
    aggregated_weights = []
    for i in range(len(model.weights)):
        # Weighted sum of weights at index i
        w_sum = np.sum([client_weights[i] * samples for client_weights, samples in results], axis=0)
        # Divide by total samples
        aggregated_weights.append(w_sum / total_samples)
    
    # Set weights to new model
    model.set_weights(aggregated_weights)
    
    new_version = version + 1
    new_path = os.path.join(MODEL_DIR, f"model_v{new_version}.keras")
    model.save(new_path)
    
    # Update DB
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO rounds (num_clients) VALUES (?)", (len(updates),))
    c.execute("INSERT INTO global_models (version, file_path) VALUES (?, ?)", (new_version, new_path))
    conn.commit()
    conn.close()
    
    print(f"Aggregation complete. New model version: {new_version}")
    round_lock = False

@app.post("/submit_update")
async def submit_update(payload: UpdatePayload, background_tasks: BackgroundTasks):
    global current_round_updates, round_lock
    
    try:
        decrypted = decrypt_payload({
            "encrypted_key": payload.encrypted_key,
            "nonce": payload.nonce,
            "ciphertext": payload.ciphertext
        }, private_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    class DecryptedPayload:
        def __init__(self, c, w, d):
            self.client_id = c
            self.weight_delta = w
            self.data_samples = d
            
    decrypted_payload = DecryptedPayload(
        payload.client_id,
        decrypted["weight_delta"],
        decrypted["data_samples"]
    )
    
    # Register client
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO clients (client_id) VALUES (?)", (payload.client_id,))
    conn.commit()
    conn.close()
    
    current_round_updates.append(decrypted_payload)
    
    if len(current_round_updates) >= MIN_CLIENTS_PER_ROUND and not round_lock:
        background_tasks.add_task(aggregate_updates)
        
    return {"status": "success", "message": f"Update received. Queue size: {len(current_round_updates)}"}

@app.get("/global_model")
def get_global_model():
    version, path = get_current_model()
    return {"version": version, "file_path": path}

@app.get("/score/{user_id}")
def get_score(user_id: str, model_output: float = Query(..., description="The locally computed model output")):
    # Record the score and return it
    print(f"Received AltScore for {user_id}: {model_output}")
    return {"user_id": user_id, "altscore": model_output, "status": "recorded"}

@app.get("/public_key")
def get_public_key():
    return {"public_key_pem": load_public_key_pem("keys")}
