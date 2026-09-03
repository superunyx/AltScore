import os
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import base64
import json

def generate_server_keypair(key_dir):
    os.makedirs(key_dir, exist_ok=True)
    priv_path = os.path.join(key_dir, "server_private.pem")
    pub_path = os.path.join(key_dir, "server_public.pem")
    if os.path.exists(priv_path) and os.path.exists(pub_path):
        return
    
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    
    with open(priv_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
    os.chmod(priv_path, 0o600)
    
    public_key = private_key.public_key()
    with open(pub_path, "wb") as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))

def load_public_key_pem(key_dir):
    with open(os.path.join(key_dir, "server_public.pem"), "rb") as f:
        return f.read().decode('utf-8')

def load_private_key(key_dir):
    with open(os.path.join(key_dir, "server_private.pem"), "rb") as f:
        return serialization.load_pem_private_key(
            f.read(),
            password=None,
        )

def encrypt_payload(plaintext_obj: dict, server_public_key_pem: str) -> dict:
    public_key = serialization.load_pem_public_key(server_public_key_pem.encode('utf-8'))
    
    aes_key = AESGCM.generate_key(bit_length=256)
    aesgcm = AESGCM(aes_key)
    nonce = os.urandom(12)
    
    plaintext = json.dumps(plaintext_obj).encode('utf-8')
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    
    encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    return {
        "encrypted_key": base64.b64encode(encrypted_key).decode('utf-8'),
        "nonce": base64.b64encode(nonce).decode('utf-8'),
        "ciphertext": base64.b64encode(ciphertext).decode('utf-8')
    }

def decrypt_payload(envelope: dict, private_key) -> dict:
    encrypted_key = base64.b64decode(envelope["encrypted_key"])
    nonce = base64.b64decode(envelope["nonce"])
    ciphertext = base64.b64decode(envelope["ciphertext"])
    
    try:
        aes_key = private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    except Exception as e:
        raise ValueError("Failed to decrypt AES key") from e
        
    aesgcm = AESGCM(aes_key)
    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    except Exception as e:
        raise ValueError("Failed to decrypt or authenticate payload") from e
        
    return json.loads(plaintext.decode('utf-8'))
