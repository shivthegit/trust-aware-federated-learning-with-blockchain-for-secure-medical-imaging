# ================= IMPORTS =================
import torch
import hashlib
import os

from utils import load_dataset, get_loader
from train_local import SimpleCNN, train_one_epoch

# ================= CONFIG =================
DATA_PATH = "../data/chest_xray"
DEVICE = "cpu"

MODEL_SAVE_PATH = "../models/hospital1_model.pt"


# ================= HASH FUNCTION =================
def hash_model(model):
    model_bytes = str(model.state_dict()).encode()
    return hashlib.sha256(model_bytes).hexdigest()


# ================= MAIN =================
def main():

    # ⭐ create models folder automatically
    os.makedirs("../models", exist_ok=True)

    print("\n==============================")
    print("HOSPITAL A LOCAL TRAINING STARTED")
    print("==============================\n")

    # 1️⃣ Load dataset
    train_data, _ = load_dataset(DATA_PATH)

    # 2️⃣ Create dataloader
    train_loader = get_loader(
        train_data,
        batch_size=16,
        shuffle=True
    )

    # 3️⃣ Create model
    model = SimpleCNN()

    # 4️⃣ Local training
    loss, acc = train_one_epoch(
        model,
        train_loader,
        device=DEVICE
    )

    # 5️⃣ Generate model hash
    model_hash = hash_model(model)

    # 6️⃣ Save local model
    torch.save(
        model.state_dict(),
        MODEL_SAVE_PATH
    )

    print("\nTRAINING COMPLETE")
    print(f"Loss      : {loss:.4f}")
    print(f"Accuracy  : {acc:.2f}%")
    print(f"Hash      : {model_hash[:15]}...")

    print("\nMODEL SAVED SUCCESSFULLY")
    print("Path:", MODEL_SAVE_PATH)

    print("\nNEXT STEP:")
    print("Upload hospital1_model.pt to Google Drive shared folder")


# ================= RUN =================
if __name__ == "__main__":
    main()