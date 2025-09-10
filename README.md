# 🏎️ AI Car Racing Bot  

This project implements an **AI-powered car racing bot** using **LSTM** (Long Short-Term Memory networks) trained with **imitation learning**.  
The model learns to mimic human driving behavior from a dataset of **my own recorded gameplay**.

---

## 🚀 Features
- 🧠 **Imitation learning** with LSTM (mimics human driving).  
- 🔗 **ONNX Runtime** for serving the trained model in Node.js.  
- 🌐 Real-time **WebSocket server** for AI + player or multiplayer interaction.  
- 🎮 Custom **3D car racing game built with Three.js**.  
- ✋ **Handtracking Steering**: Control the car using hand gestures via webcam.  
  👉 More info on gestures here: [Handtracking Steering Wheel Repo](https://github.com/ArnavJoshi14/handtracking-steering-wheel)  
- 🔮 Future plan: **Reinforcement learning** for self-improving AI and better race tracks.  

---

## 📂 Project Structure
```
.
├── ai_bot/                  # AI model & training
│   ├── train_lstm.py        # Training script (imitation learning)
│   ├── torch_to_onnx.py     # Convert PyTorch → ONNX
│   ├── lstm_model.onnx      # Trained model (ignored in .gitignore)
│   ├── requirements.txt     # Python dependencies
│   └── venv/                # Python virtual environment (ignored)
│
├── handtracking-steering/   # Steer virtually using your hands
│   ├── requirements.txt     # Python dependencies
│   ├── directkeys.py        # Simulate key presses
│   └── main.py              # Steer virtually using hands
│
├── public/                  
│   ├── models/              # Frontend assets for the racing game
│   └── index.html           # Game UI
│
├── server.js                # Node.js backend server (WebSocket + ONNX)
├── package.json             # Node.js dependencies       
├── .gitignore               # Ignore large / generated files
└── README.md                
```

---

## ⚙️ Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ArnavJoshi14/websocket-multiplayer-car-racing.git
cd websocket-multiplayer-car-racing
```

---

### 2. Install Python Environment (for training / conversion)
```bash
cd ai_bot
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

pip install -r requirements.txt
```

If you already have a trained model, skip training.

---

## 3. Training the Model
```bash
cd ai_bot
python train_lstm.py
python torch_to_onnx.py
```
This will export `lstm_model.onnx`.

---

### 4. Install Node.js Dependencies
```bash
npm install
```

---

## 5. Running the Server
Start the backend:
```bash
node server.js
```

The WebSocket server will:
- Wait for **player connections**  
- Run the **AI car** using the ONNX model  
- Sync race states between human and AI  

---

## 📦 Requirements

### Python
- torch  
- numpy  
- onnx  
- onnxruntime  

### Node.js
- express  
- ws  
- onnxruntime-node  

---

## 📝 Notes
- This is currently an **imitation learning model** trained on a **limited dataset of my own driving**.  
- The AI performance is basic and will be **improved with reinforcement learning** in future versions.  
- The racing game is powered by **Three.js** for rendering.  
- Handtracking steering lets you control the car using **your webcam and gestures**. More info in the [handtracking repo](https://github.com/ArnavJoshi14/handtracking-steering-wheel).  

---

## 📌 Roadmap
- [ ] Improve dataset (collect more driving data).
- [ ] Transition to **reinforcement learning** for autonomous self-improvement.  
- [ ] Add multiplayer support with multiple AI cars.  
- [ ] Expand race tracks and improve game physics.  

---

## 🎨 Credits (3D Car Model)
- **Model + Textures** by: **Spaehling**  
- On behalf of: [Grip420](https://www.grip420.com/)  