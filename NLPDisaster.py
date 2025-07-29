
import os
import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Parameters
BATCH_SIZE = 32
EPOCHS = 3
MAX_LEN = 128

# Load data
df = pd.read_csv("train.csv")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["target"].tolist(), test_size=0.2, random_state=42
)

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")

class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenize(texts)
        self.labels = labels

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.float)
        }

    def __len__(self):
        return len(self.labels)

train_dataset = TweetDataset(train_texts, train_labels)
val_dataset = TweetDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)

# Model
class BERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)

model = BERTClassifier().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
scaler = GradScaler()

# Training
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        with autocast():
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

# Prediction
test_df = pd.read_csv("test.csv")
test_encodings = tokenize(test_df["text"].tolist())
test_dataset = TweetDataset(test_df["text"].tolist(), [0]*len(test_df))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)

model.eval()
predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        with autocast():
            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            predictions.extend(preds)

submission = pd.read_csv("sample_submission.csv")
submission["target"] = predictions
submission.to_csv("submission.csv", index=False)
print("Saved predictions to submission.csv")
