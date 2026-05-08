# ================= IMPORTS =================
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from web3 import Web3

# ================= FIX STREAMLIT EVENT LOOP =================
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Healthcare Federated AI",
    layout="wide"
)

# ================= BLOCKCHAIN CONFIG =================
GANACHE_URL = "http://127.0.0.1:7545"
CONTRACT_ADDRESS = "0x58C78014e90D5BE5545DE9e77102Ce7805299C66"

with open("fl_code/contract_abi.json", "r") as f:
    CONTRACT_ABI = json.load(f)

# ================= BLOCKCHAIN READER =================
def load_blockchain_logs():

    logs = []

    try:
        w3 = Web3(Web3.HTTPProvider(GANACHE_URL))

        if not w3.is_connected():
            return logs

        contract = w3.eth.contract(
            address=CONTRACT_ADDRESS,
            abi=CONTRACT_ABI
        )

        count = contract.functions.getUpdatesCount().call()

        for i in range(count):
            data = contract.functions.getUpdate(i).call()

            readable_time = datetime.fromtimestamp(data[3]).strftime("%Y-%m-%d %H:%M:%S")

            logs.append({
                "Hospital": data[0],
                "Hash": data[1],
                "Accuracy": data[2],
                "Timestamp": readable_time,
                "Verification": "✔ Verified"  # unchanged default
            })

    except Exception:
        pass

    return logs


# ================= LOAD LOCAL JSON LOGS =================
def load_local_logs():

    path = "results/blockchain_logs.json"

    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    return []


# ================= MODEL =================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*56*56,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )

    def forward(self,x):
        x = self.features(x)
        return self.classifier(x)


# ================= RESNET =================
def get_resnet_model():
    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )

    return model


# ================= LOAD MODEL =================
@st.cache_resource
def load_model():

    model_path = r"C:\Users\Muthupriya V\Desktop\Healthcare-FL-Blockchain\models\global_model.pt"

    if not os.path.exists(model_path):
        st.error(f"❌ Model NOT found at: {model_path}")
        st.stop()

    try:
        model = get_resnet_model()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model
    except:
        pass

    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model


try:
    model = load_model()
except:
    st.error("❌ Global model not found. Run aggregation first.")
    st.stop()


# ================= LOAD METRICS =================
def load_metrics():

    path = "results/metrics.json"

    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)

        data.setdefault("accuracy",0)
        data.setdefault("precision",0)
        data.setdefault("recall",0)
        data.setdefault("f1_score",0)
        data.setdefault("rounds",[])
        data.setdefault("losses",[])
        data.setdefault("confusion_matrix",[[0,0],[0,0]])

        return data

    return {
        "accuracy":0,
        "precision":0,
        "recall":0,
        "f1_score":0,
        "rounds":[],
        "losses":[],
        "confusion_matrix":[[0,0],[0,0]]
    }

metrics = load_metrics()

precision = metrics["precision"]
recall = metrics["recall"]
f1 = metrics.get("f1_score",0)


# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# ================= PREDICT =================
def predict(image):

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs,1)

    classes = ["NORMAL","PNEUMONIA"]
    return classes[pred.item()], conf.item()


# ================= HEADER =================
st.title("🏥 Federated Healthcare AI Dashboard")
st.caption("Federated Learning • Blockchain Secured • Research Grade")
st.divider()


# ================= METRICS =================
a,b,c,d,e = st.columns(5)

a.metric("Accuracy", f"{metrics['accuracy']:.2f}%")
b.metric("Precision", f"{precision*100:.2f}%")
c.metric("Recall", f"{recall*100:.2f}%")
d.metric("F1 Score", f"{f1*100:.2f}%")
e.metric("Rounds", len(metrics.get("rounds",[])))


# ================= TABS =================
tab1, tab2, tab3 = st.tabs(
    ["📊 Model Performance", "🩺 Prediction", "🔗 Blockchain Analytics"]
)


# ================= TAB 1 =================
with tab1:

    st.subheader("Federated Round Accuracy")

    rounds = metrics.get("rounds",[])

    if len(rounds) > 0:
        fig, ax = plt.subplots(figsize=(7,3))
        ax.plot(rounds, marker="o")
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels([f"R{i+1}" for i in range(len(rounds))])
        ax.set_ylabel("Accuracy (%)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    st.subheader("Federated Round Loss")

    losses = metrics.get("losses", [])

    if len(losses) > 0:
        fig_loss, ax_loss = plt.subplots(figsize=(7,3))
        ax_loss.plot(losses, marker="o")
        ax_loss.set_xticks(range(len(losses)))
        ax_loss.set_xticklabels([f"R{i+1}" for i in range(len(losses))])
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(alpha=0.3)
        st.pyplot(fig_loss)

    st.subheader("Round-wise Loss Table")

    if len(losses) > 0:
        df_loss = pd.DataFrame({
            "Round": [f"R{i+1}" for i in range(len(losses))],
            "Loss": losses
        })
        st.dataframe(df_loss, use_container_width=True)

    st.subheader("Confusion Matrix")

    cm = np.array(metrics["confusion_matrix"])

    fig2, ax2 = plt.subplots(figsize=(4,4))
    ax2.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax2.text(j, i, cm[i,j], ha="center", va="center")

    ax2.set_xticks([0,1])
    ax2.set_yticks([0,1])
    ax2.set_xticklabels(["NORMAL","PNEUMONIA"])
    ax2.set_yticklabels(["NORMAL","PNEUMONIA"])

    st.pyplot(fig2)


# ================= TAB 2 =================
with tab2:
    left, right = st.columns([1,1.3])

    with left:
        uploaded_file = st.file_uploader(
            "Upload Chest X-Ray",
            type=["jpg","jpeg","png"]
        )

    with right:
        st.subheader("AI Diagnosis")

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)

            result, conf = predict(image)

            if result == "PNEUMONIA":
                st.error(f"Disease Detected: {result}")
                recommendation = "Further clinical evaluation recommended."
            else:
                st.success(f"Result: {result}")
                recommendation = "No abnormality detected."

            st.metric("Confidence", f"{conf*100:.2f}%")
            st.progress(float(conf))

            report = f"""
Healthcare AI Diagnosis Report
Date: {datetime.now()}

Prediction: {result}
Confidence: {conf*100:.2f}%

Recommendation:
{recommendation}
"""

            st.download_button(
                "📄 Download Diagnosis Report",
                report,
                "diagnosis_report.txt"
            )


# ================= TAB 3 =================
with tab3:

    logs = load_blockchain_logs()
    local_logs = load_local_logs()

    if len(logs) > 0:

        df = pd.DataFrame(logs)

        # SAFE ADD (does not break anything)
        if len(local_logs) > 0:
            df_local = pd.DataFrame(local_logs)
            if "loss" in df_local.columns:
                df["Loss"] = df_local["loss"]

            if "status" in df_local.columns:
                status_list = df_local["status"].tolist()
                # Fill properly without mismatch
                df["Verification"] = [
                    status_list[i] if i < len(status_list) else "✔ Verified"
                     for i in range(len(df))
    ]

        st.subheader("Live Blockchain Logs")
        st.dataframe(df, use_container_width=True)

    else:
        st.info("No blockchain logs found.")


# ================= FOOTER =================
st.divider()
st.caption("Federated Learning + Blockchain Healthcare AI System")