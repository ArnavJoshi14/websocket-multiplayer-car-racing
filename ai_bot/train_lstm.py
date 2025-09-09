import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

SEQ_LEN = 20
EPOCHS = 20
BATCH_SIZE = 64
MODEL_OUT = "lstm_model.pth"
JSONL_PATH = "../logs/game_logs.jsonl"

# load dataset
def load_jsonl(path):
    states, actions = [], []
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("type") != "log":
                continue
            s = entry["state"]
            a = entry["action"]
            state_vec = [s["posX"], s["posY"], s["posZ"], s["speed"], s["rot"]]
            action_vec = [int(a["up"]), int(a["down"]), int(a["left"]), int(a["right"])]
            states.append(state_vec)
            actions.append(action_vec)
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32)

print("Loading dataset...")
states, actions = load_jsonl(JSONL_PATH)
print(f"Loaded: ", states.shape, actions.shape)

# build sequences
X, y = [], []
for i in range(len(states) - SEQ_LEN):
    X.append(states[i : i + SEQ_LEN])
    y.append(actions[i + SEQ_LEN])
X = np.array(X)
y = np.array(y)
print("Built sequences: ", X.shape, y.shape)

# train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

# torch dataset
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# model
class LSTMModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take last time step
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out

state_dim = X.shape[2]
action_dim = y.shape[1]
model = LSTMModel(state_dim, action_dim)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# training loop
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += ((preds > 0.5) == (yb > 0.5)).all(dim=1).sum().item()
        total += yb.size(0)
    train_acc = correct / total

    # validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            val_loss += criterion(preds, yb).item()
            val_correct += ((preds > 0.5) == yb.bool()).all(dim=1).sum().item()
            val_total += yb.size(0)
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")

# save model
torch.save(model.state_dict(), MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")

example_input = torch.randn(1, SEQ_LEN, state_dim)
traced = torch.jit.trace(model, example_input)
traced.save("lstm_model.pt") # TorchScript model
print("TorchScript model saved to lstm_model.pt")