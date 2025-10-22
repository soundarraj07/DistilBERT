# ============================================================
# ✈️ DistilBERT Intent Classification + Firebase + Streamlit
# ============================================================
import streamlit as st
import torch
import re, string
import firebase_admin
from firebase_admin import credentials, firestore
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Firebase Setup (Safe Init)
# -----------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate("distilbert-20fc6-firebase-adminsdk-fbsvc-acf3cd9f78.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# -----------------------------
# Load Model and Tokenizer
# -----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("./distilbert_airline")
model = DistilBertForSequenceClassification.from_pretrained("./distilbert_airline")

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\{\{.*?\}\}', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------
# Predict Intent
# -----------------------------
def predict_intent(text):
    text = preprocess(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = torch.argmax(logits, dim=1).item()
    label = model.config.id2label.get(pred_id, f"label_{pred_id}")
    return label

# -----------------------------
# Retraining Function
# -----------------------------
def retrain_model():
    corrections_ref = db.collection("human_corrections").where("processed_for_retrain", "==", False)
    corrections = list(corrections_ref.stream())

    if len(corrections) < 5:  # retrain after 5+ corrections
        return

    records = [c.to_dict() for c in corrections]
    texts = [r["user_text"] for r in records if "corrected_intent" in r]
    labels = [r["corrected_intent"] for r in records if "corrected_intent" in r]

    if not texts or not labels:
        return

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    dataset = Dataset.from_dict({"instruction": texts, "label": labels_encoded})

    def tokenize(batch):
        return tokenizer(batch["instruction"], padding="max_length", truncation=True, max_length=128)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Fine-tune model on new corrections
    training_args = TrainingArguments(
        output_dir="./distilbert_airline",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        save_total_limit=1,
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()

    # Save updated model
    model.save_pretrained("./distilbert_airline")
    tokenizer.save_pretrained("./distilbert_airline")

    # Mark corrections as processed
    for c in corrections:
        db.collection("human_corrections").document(c.id).update({"processed_for_retrain": True})

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("✈️ Airline Intent Classifier")

menu = st.sidebar.selectbox("Choose Mode:", ["User", "Human Agent"])

# -----------------------------
# User Mode
# -----------------------------
if menu == "User":
    st.subheader("🧑‍💻 User Input")
    user_text = st.text_area("Enter your message:")

    if st.button("Predict Intent"):
        if not user_text.strip():
            st.warning("Please enter a message first!")
        else:
            pred_intent = predict_intent(user_text)
            st.success(f"**Predicted Intent:** {pred_intent}")

            is_correct = st.radio("Is the prediction correct?", ["Yes", "No"], key="feedback")

            if st.button("Submit Feedback"):
                if is_correct == "Yes":
                    db.collection("user_feedback").add({
                        "user_text": user_text,
                        "predicted_intent": pred_intent,
                        "is_correct": True,
                        "timestamp": firestore.SERVER_TIMESTAMP
                    })
                    st.info("✅ Feedback saved: prediction correct.")
                else:
                    db.collection("human_corrections").add({
                        "user_text": user_text,
                        "predicted_intent": pred_intent,
                        "processed_for_retrain": False,
                        "timestamp": firestore.SERVER_TIMESTAMP
                    })
                    st.warning("❗ Sent for human correction.")
                retrain_model()

# -----------------------------
# Human Agent Mode
# -----------------------------
elif menu == "Human Agent":
    st.subheader("🧍 Human Agent Dashboard")
    corrections = db.collection("human_corrections").where("processed_for_retrain", "==", False).stream()
    found = False
    for doc in corrections:
        found = True
        data = doc.to_dict()
        st.write(f"**User Text:** {data['user_text']}")
        st.write(f"**Predicted Intent:** {data['predicted_intent']}")
        corrected_intent = st.text_input("Enter Correct Intent", value=data["predicted_intent"], key=f"in_{doc.id}")
        if st.button("Submit Correction", key=f"btn_{doc.id}"):
            db.collection("human_corrections").document(doc.id).update({
                "corrected_intent": corrected_intent
            })
            st.success("✅ Correction saved!")
            retrain_model()
    if not found:
        st.info("No pending corrections right now 🎉")

