"""
====================================================================
PROJECT AEGIS — Proprietary Vault & Intellectual Property Shield
====================================================================
Encrypts proprietary trading models, neural consensus logic, and
genetic evolver into an AES-256 encrypted payload (data/vault_core.enc).

Even if your GitHub repository is public, competitors CANNOT read
or steal your business model without your private AEGIS_VAULT_KEY.
====================================================================
"""

import os
import sys
import json
import base64
import argparse
from cryptography.fernet import Fernet

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
VAULT_FILE = os.path.join(DATA_DIR, "vault_core.enc")
KEY_FILE = os.path.join(DATA_DIR, ".vault_key")

# The 4 core proprietary intellectual property files
PROPRIETARY_FILES = [
    "strategy_engine.py",
    "neuro_voter.py",
    "trading_brain.py",
    "genetic_evolver.py",
]


def generate_key() -> str:
    """Generate a new cryptographically secure AES-256 key."""
    key = Fernet.generate_key().decode()
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(KEY_FILE, "w", encoding="utf-8") as f:
        f.write(key.strip())
    print(f"\nSUCCESS: New AES-256 Vault Key Generated!\n")
    print(f"================================================================")
    print(f"YOUR PRIVATE VAULT KEY (KEEP THIS SECRET):")
    print(f"  {key}")
    print(f"================================================================")
    print(f"\nAdd this key to your GitHub Secrets as: AEGIS_VAULT_KEY")
    return key


def get_key() -> bytes:
    """Retrieve key from environment variable or local .vault_key."""
    key_str = os.getenv("AEGIS_VAULT_KEY", "")
    if not key_str and os.path.exists(KEY_FILE):
        try:
            with open(KEY_FILE, "r", encoding="utf-8") as f:
                key_str = f.read().strip()
        except Exception:
            pass
    if not key_str:
        raise ValueError("Missing AEGIS_VAULT_KEY! Provide via env var or generate using --generate-key")
    return key_str.encode()


def pack_vault():
    """Pack and encrypt all proprietary files into data/vault_core.enc."""
    key = get_key()
    fernet = Fernet(key)

    bundle = {}
    for filename in PROPRIETARY_FILES:
        path = os.path.join(SRC_DIR, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                bundle[filename] = f.read()
            print(f"  [+] Protected: {filename} ({os.path.getsize(path)} bytes)")
        else:
            print(f"  [-] Warning: {filename} not found")

    raw_json = json.dumps(bundle).encode("utf-8")
    encrypted_payload = fernet.encrypt(raw_json)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(VAULT_FILE, "wb") as f:
        f.write(encrypted_payload)

    print(f"\nSUCCESS: Vault packed! Encrypted payload saved to: data/vault_core.enc ({len(encrypted_payload)} bytes)")


def unlock_vault():
    """Decrypt and unpack all proprietary files for execution in memory/disk."""
    key = get_key()
    fernet = Fernet(key)

    if not os.path.exists(VAULT_FILE):
        raise FileNotFoundError(f"Vault file not found at {VAULT_FILE}")

    with open(VAULT_FILE, "rb") as f:
        encrypted_payload = f.read()

    decrypted_bytes = fernet.decrypt(encrypted_payload)
    bundle = json.loads(decrypted_bytes.decode("utf-8"))

    for filename, content in bundle.items():
        path = os.path.join(SRC_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  [+] Decrypted: {filename}")

    print("\nSUCCESS: Vault unlocked and ready for execution!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Aegis Vault & IP Shield")
    parser.add_argument("--generate-key", action="store_true", help="Generate a new AES-256 key")
    parser.add_argument("--pack", action="store_true", help="Encrypt proprietary code into data/vault_core.enc")
    parser.add_argument("--unlock", action="store_true", help="Decrypt proprietary code using AEGIS_VAULT_KEY")

    args = parser.parse_args()
    if args.generate_key:
        generate_key()
    elif args.pack:
        pack_vault()
    elif args.unlock:
        unlock_vault()
    else:
        parser.print_help()