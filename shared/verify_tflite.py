import tensorflow as tf
import numpy as np
import os

TFLITE_MODEL_PATH = "base_model.tflite"

def main():
    if not os.path.exists(TFLITE_MODEL_PATH):
        print(f"Error: {TFLITE_MODEL_PATH} not found.")
        return

    print(f"Loading {TFLITE_MODEL_PATH}...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    
    signatures = interpreter.get_signature_list()
    print("Available Signatures:", signatures)
    
    # Check that all expected signatures are present
    expected_signatures = {'train', 'infer', 'save', 'restore'}
    if not expected_signatures.issubset(set(signatures.keys())):
        print(f"Warning: Missing signatures. Expected {expected_signatures}, found {set(signatures.keys())}")
    
    # 1. Save (saves whatever state, possibly uninitialized/default)
    print("\n--- Verifying 'save' signature ---")
    save_fn = interpreter.get_signature_runner('save')
    checkpoint_path = np.array(b"test_checkpoint.ckpt")
    save_result = save_fn(checkpoint_path=checkpoint_path)
    print(f"Save completed. Checkpoint index exists: {os.path.exists('test_checkpoint.ckpt.index')}")
    
    # 2. Restore
    print("\n--- Verifying 'restore' signature ---")
    restore_fn = interpreter.get_signature_runner('restore')
    restore_result = restore_fn(checkpoint_path=checkpoint_path)
    print("Restore completed successfully.")

    # 3. Infer
    print("\n--- Verifying 'infer' signature ---")
    infer_fn = interpreter.get_signature_runner('infer')
    dummy_x = np.random.random((1, 30, 3)).astype(np.float32)
    infer_result = infer_fn(x=dummy_x)
    print(f"Infer result output shape: {infer_result['output'].shape}")
    print(f"Infer result value: {infer_result['output'][0][0]:.4f}")
    
    # 4. Train
    print("\n--- Verifying 'train' signature ---")
    train_fn = interpreter.get_signature_runner('train')
    dummy_y = np.array([[0.8]], dtype=np.float32)
    train_result = train_fn(x=dummy_x, y=dummy_y)
    print(f"Train loss after step: {train_result['loss']:.4f}")
    
    print("\n✅ All on-device TFLite signatures verified successfully!")
    
    # Cleanup dummy checkpoint
    for ext in ['.index', '.data-00000-of-00001']:
        if os.path.exists(f"test_checkpoint.ckpt{ext}"):
            os.remove(f"test_checkpoint.ckpt{ext}")

if __name__ == "__main__":
    main()
