import os
import json
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf

DATA_DIR = "../data/generated_users"
MODEL_SAVE_PATH = "base_model.keras"
TFLITE_SAVE_PATH = "base_model.tflite"

import sys
sys.path.append(os.path.dirname(__file__))
from feature_engineering import compute_ratio_features

def load_data():
    X = []
    y = []
    
    if not os.path.exists(DATA_DIR):
        raise ValueError(f"Data directory {DATA_DIR} not found.")
        
    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith('.json'):
            continue
            
        with open(os.path.join(DATA_DIR, filename), 'r') as f:
            data = json.load(f)
            
        score = data['reliability_score']
        app_usage = {item['date']: item for item in data['app_usage']}
        sms_logs = data.get('sms_logs', [])
                
        if not app_usage:
            continue
            
        min_date_str = min(app_usage.keys())
        min_date = datetime.strptime(min_date_str, "%Y-%m-%d")
        
        for start_offset in range(0, 61, 5): # 13 windows per user
            window_start = min_date + timedelta(days=start_offset)
            window_end = window_start + timedelta(days=30)
            
            features = compute_ratio_features(sms_logs, app_usage, window_start, window_end)
            X.append(features)
            y.append(score)
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def build_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(8,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

class OnDeviceModel(tf.Module):
    """
    Wrapper model to provide specific signatures required for
    on-device training with TFLite.
    """
    def __init__(self, keras_model):
        super(OnDeviceModel, self).__init__()
        self.model = keras_model
        # Use a much lower learning rate for local fine-tuning on tiny data
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)

    @tf.function(input_signature=[
        tf.TensorSpec([None, 8], tf.float32),
        tf.TensorSpec([None, 1], tf.float32)
    ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            prediction = self.model(x, training=True)
            loss = tf.keras.losses.MeanSquaredError()(y, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return {"loss": loss}

    @tf.function(input_signature=[tf.TensorSpec([None, 8], tf.float32)])
    def infer(self, x):
        return {"output": self.model(x, training=False)}

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [tf.identity(weight) for weight in self.model.weights]
        with tf.device('/cpu:0'):
            tf.raw_ops.Save(
                filename=checkpoint_path, tensor_names=tensor_names,
                data=tensors_to_save, name='save')
        return {"filename": checkpoint_path}

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        with tf.device('/cpu:0'):
            for weight in self.model.weights:
                restored = tf.raw_ops.Restore(
                    file_pattern=checkpoint_path, tensor_name=weight.name, dt=weight.dtype,
                    name='restore')
                restored = tf.ensure_shape(restored, weight.shape)
                weight.assign(restored)
        return {}

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
    def export_weights(self, dummy):
        return {weight.name: tf.identity(weight) for weight in self.model.weights}

def main():
    print("Loading synthetic data from Phase 1...")
    X, y = load_data()
    print(f"Loaded {len(X)} user records. Shape: {X.shape}")
    
    # Basic train/test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print("\nBuilding and pre-training base model...")
    model = build_keras_model()
    # Simple training loop over the synthetic data
    model.fit(X_train, y_train, epochs=25, batch_size=4, validation_split=0.1, verbose=1)
    
    print("\nEvaluating on held-out test set...")
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"--> Test Loss (MSE): {loss:.4f}")
    print(f"--> Test MAE: {mae:.4f}")
    
    print(f"\nSaving standard Keras model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    
    print("Converting to TFLite format with on-device training signature...")
    tflite_module = OnDeviceModel(model)
    
    tf.saved_model.save(
        tflite_module, 
        "saved_model_dir",
        signatures={
            'train': tflite_module.train.get_concrete_function(),
            'infer': tflite_module.infer.get_concrete_function(),
            'save': tflite_module.save.get_concrete_function(),
            'restore': tflite_module.restore.get_concrete_function(),
            'export_weights': tflite_module.export_weights.get_concrete_function(),
        }
    )
    
    converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_dir")
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()
    
    with open(TFLITE_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)
        
    print(f"Saved TFLite model to {TFLITE_SAVE_PATH}.")
    print("Base Model V1.0 is ready for deployment!")

if __name__ == "__main__":
    main()
