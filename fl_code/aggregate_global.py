# ================= IMPORTS =================
import torch
import os
import json
import hashlib
import matplotlib.pyplot as plt
from datetime import datetime

from web3 import Web3

from train_local import get_model, train_one_epoch, evaluate
from fedavg import fedavg
from utils import load_dataset, get_loader
from split_hospitals import split_into_hospitals


# ================= ATTACK =================
def poison_model(model):
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    return model


# ================= CONFIG =================
DEVICE = "cpu"
DATA_PATH = "../data/chest_xray"

MODELS_DIR = "../models"
RESULTS_DIR = "../results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_ROUNDS = 10
LOCAL_EPOCHS = 2


# ================= TRUST =================
client_trust = {
    "H1": 1.0,
    "H2": 1.0
}


# ================= BLOCKCHAIN =================
GANACHE_URL = "http://127.0.0.1:7545"

CONTRACT_ADDRESS = "0x58C78014e90D5BE5545DE9e77102Ce7805299C66"
ACCOUNT_ADDRESS = "0x973354284dEd7a59DA6D97594d80039283fdFb20"
PRIVATE_KEY = "0x00a0ec91cf8995092c3a5b88d2029ebae705089dcfde61ed97b159a8b3d07524"


with open("contract_abi.json", "r") as f:
    CONTRACT_ABI = json.load(f)


# ================= HASH =================
def hash_model(model):
    model_bytes = str(model.state_dict()).encode()
    return hashlib.sha256(model_bytes).hexdigest()


# ================= LOG =================
def log_to_blockchain(tagged_hash, accuracy):

    print(f"\nLogging: {tagged_hash[:40]}...")

    w3 = Web3(Web3.HTTPProvider(GANACHE_URL))

    if not w3.is_connected():
        return

    contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)

    nonce = w3.eth.get_transaction_count(ACCOUNT_ADDRESS)

    tx = contract.functions.submitUpdate(
        tagged_hash,
        int(accuracy)
    ).build_transaction({
        "from": ACCOUNT_ADDRESS,
        "nonce": nonce,
        "gas": 2000000,
        "gasPrice": w3.to_wei("20", "gwei")
    })

    signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)

    print("✔ Blockchain log added!")
    print("TX HASH:", tx_hash.hex())


# ================= VERIFY =================
def verify_from_blockchain(expected_hash):

    w3 = Web3(Web3.HTTPProvider(GANACHE_URL))

    if not w3.is_connected():
        return False

    contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)

    total = contract.functions.getUpdatesCount().call()

    if total == 0:
        return False

    last_update = contract.functions.getUpdate(total - 1).call()

    stored_hash = last_update[1]
    timestamp = last_update[3]

    stored_clean_hash = stored_hash.split("_")[-1]

    readable_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    print("Stored Hash:", stored_hash[:40])
    print("Local Hash :", expected_hash[:40])
    print("Timestamp  :", readable_time)

    if expected_hash == stored_clean_hash:
        print("✔ Model VERIFIED")
        return True
    else:
        print("❌ Model TAMPERED")
        return False


# ================= TRUST-WEIGHTED FEDAVG =================
def trust_weighted_fedavg(local_models, weights):
    avg_state = {}

    total_weight = sum(weights)

    for key in local_models[0].keys():
        avg_state[key] = sum(
            local_models[i][key] * (weights[i] / total_weight)
            for i in range(len(local_models))
        )

    return avg_state


# ================= MAIN =================
def main():

    train_data, test_data = load_dataset(DATA_PATH)
    test_loader = get_loader(test_data, batch_size=32, shuffle=False)

    hospitals = split_into_hospitals(train_data, num_hospitals=2)

    global_model = get_model()

    best_acc = 0
    best_model_state = None
    best_metrics = {}

    metrics_data = {
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1_score": 0,
        "rounds": [],
        "losses": [],
        "confusion_matrix": [[0, 0], [0, 0]]
    }

    # 🔥 NEW: blockchain_logs.json storage
    blockchain_logs = []

    for r in range(NUM_ROUNDS):

        print(f"\nROUND {r+1}")

        current_lr = 0.0005 * (0.9 ** r)

        local_states = []
        trust_weights = []

        for i, hospital_data in enumerate(hospitals):

            is_attacker = (i == 1 and (2 <= r <= 3 or r == 6))

            print(f"Hospital {i+1} training...")

            loader = get_loader(hospital_data, batch_size=32, shuffle=True)

            local_model = get_model()
            local_model.load_state_dict(global_model.state_dict())

            for _ in range(LOCAL_EPOCHS):
                loss, acc = train_one_epoch(
                    local_model,
                    loader,
                    device=DEVICE,
                    lr=current_lr
                )

            print(f"Loss: {loss:.4f} | Acc: {acc:.2f}%")

            original_hash = hash_model(local_model)
            tag = f"H{i+1}_R{r+1}_{original_hash}"

            log_to_blockchain(tag, acc)

            if is_attacker:
                print("⚠️ ATTACKER ACTIVE")
                local_model = poison_model(local_model)

            new_hash = hash_model(local_model)
            is_verified = verify_from_blockchain(new_hash)

            # 🔥 NEW: status
            status = "✔ VERIFIED" if is_verified else "❌ TAMPERED"

            # 🔥 NEW: save log
            blockchain_logs.append({
                "hospital": f"H{i+1}",
                "round": r+1,
                "accuracy": acc,
                "status": status
            })

            hospital_id = f"H{i+1}"

            if is_verified:
                client_trust[hospital_id] += 0.1
            else:
                client_trust[hospital_id] -= 0.5

            print("Trust:", client_trust)

            if client_trust[hospital_id] < 0.3:
                print(f"🚫 Skipping Hospital {i+1}")
                continue

            local_states.append(local_model.state_dict())
            trust_weights.append(max(client_trust[hospital_id], 0.01))

        if len(local_states) == 0:
            continue

        avg_state = trust_weighted_fedavg(local_states, trust_weights)
        global_model.load_state_dict(avg_state)

        global_acc, precision, recall, f1, cm = evaluate(
            global_model, test_loader, device=DEVICE
        )

        print(f"\n🌍 Global Accuracy: {global_acc:.2f}%")

        metrics_data["rounds"].append(global_acc)
        metrics_data["losses"].append(loss)

        if global_acc > best_acc:
            best_acc = global_acc
            best_model_state = global_model.state_dict()

            best_metrics = {
                "accuracy": global_acc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": cm
            }

        global_hash = hash_model(global_model)
        log_to_blockchain(f"GLOBAL_R{r+1}_{global_hash}", global_acc)
        verify_from_blockchain(global_hash)

    torch.save(best_model_state, f"{MODELS_DIR}/global_model.pt")

    metrics_data.update(best_metrics)

    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)

    # 🔥 NEW: SAVE BLOCKCHAIN STATUS FILE
    with open(f"{RESULTS_DIR}/blockchain_logs.json", "w") as f:
        json.dump(blockchain_logs, f, indent=4)

    print(f"\n✔ BEST MODEL SAVED | Accuracy: {best_acc:.2f}%")
    print("📊 Metrics saved (BEST values)")
    print("🔗 Blockchain logs saved (with status)")


if __name__ == "__main__":
    main()