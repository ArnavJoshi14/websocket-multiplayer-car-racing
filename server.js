const express = require("express");
const path = require("path");
const bodyParser = require("body-parser");
const WebSocket = require("ws");
const ort = require("onnxruntime-node");

const app = express();
const PORT = 3000;

app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, "public")));

const http = require("http");
const server = http.createServer(app);

const wss = new WebSocket.Server({ server });

let players = {};

let aiEnabled = true;
const AI_ID = "AI_PLAYER";
let aiState = { posX: 0, posY: 0.5, posZ: 0, speed: 0, rot: 0 };
let aiHistory = [];
const SEQ_LEN = 20;
let model = null;

// === Load ONNX model ===
async function loadAiModel() {
  try {
    model = await ort.InferenceSession.create(path.join(__dirname, "ai_bot", "lstm_model.onnx"));
    console.log("✅ AI model loaded (ONNX).");
  } catch (e) {
    console.error("❌ Failed to load AI model:", e);
  }
}

// Convert model outputs to action booleans
function probsToActions(probs, thresh = 0.5) {
  return {
    up: probs[0] > thresh,
    down: probs[1] > thresh,
    left: probs[2] > thresh,
    right: probs[3] > thresh
  };
}

// Run inference with ONNX
async function aiInferAction() {
  if (!model || aiHistory.length < SEQ_LEN) return null;

  const seq = aiHistory.slice(aiHistory.length - SEQ_LEN);
  const flat = seq.flat(); // flatten sequence into 1D array
  const inputTensor = new ort.Tensor("float32", Float32Array.from(flat), [1, SEQ_LEN, seq[0].length]);

  const results = await model.run({ input: inputTensor });
  const output = results.output.data; // [up, down, left, right] probabilities
  return probsToActions(output);
}

// Physics update
function aiUpdatePhysics(actions, dt) {
  const c = aiState;
  const acceleration = 8;
  const friction = 0.99;
  const turnSpeed = 1;
  const maxSpeed = 50;

  if (actions.up) c.speed -= acceleration * dt;
  else if (actions.down) c.speed += acceleration * dt;
  else c.speed *= friction;

  c.speed = Math.max(-maxSpeed, Math.min(maxSpeed * 0.5, c.speed));
  if (Math.abs(c.speed) > 0.5) {
    const turnRate = turnSpeed * dt;
    if (actions.left) c.rot += turnRate;
    if (actions.right) c.rot -= turnRate;
  }

  const vx = Math.sin(c.rot) * c.speed * dt;
  const vz = Math.cos(c.rot) * c.speed * dt;
  c.posX += vx;
  c.posZ += vz;
}

function broadcast(obj) {
  const msg = JSON.stringify(obj);
  for (let id in players) {
    const ws = players[id];
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(msg);
    }
  }
}

let lastAiTick = Date.now();
function startAiLoop() {
  const TICK_HZ = 10;
  const TICK_MS = 1000 / TICK_HZ;
  setInterval(async () => {
    const now = Date.now();
    const dt = Math.min((now - lastAiTick) / 1000, 1 / 30);
    lastAiTick = now;

    if (aiHistory.length === 0) {
      for (let i = 0; i < SEQ_LEN; i++) {
        aiHistory.push([aiState.posX, aiState.posY, aiState.posZ, aiState.speed, aiState.rot]);
      }
    }

    const probsActions = await aiInferAction();
    const actions = probsActions || { up: true, down: false, left: false, right: false };

    aiUpdatePhysics(actions, dt);

    aiHistory.push([aiState.posX, aiState.posY, aiState.posZ, aiState.speed, aiState.rot]);
    if (aiHistory.length > SEQ_LEN * 2) aiHistory.splice(0, aiHistory.length - SEQ_LEN);

    const stateData = {
      type: "state",
      playerid: AI_ID,
      state: {
        position: { x: aiState.posX, y: aiState.posY, z: aiState.posZ },
        rotation: aiState.rot,
        speed: aiState.speed,
      },
    };
    broadcast(stateData);
  }, TICK_MS);
}

loadAiModel().then(() => {
  if (aiEnabled && model) {
    startAiLoop();
  }
});

wss.on("connection", (ws) => {
  const playerId = Date.now();
  players[playerId] = ws;
  console.log(`Player ${playerId} connected`);

  ws.send(JSON.stringify({ type: "init", id: playerId }));

  if (aiEnabled) {
    ws.send(JSON.stringify({
      type: "state",
      playerid: AI_ID,
      state: {
        position: { x: aiState.posX, y: aiState.posY, z: aiState.posZ },
        rotation: aiState.rot,
        speed: aiState.speed,
      },
    }));
  }

  ws.on("message", (message) => {
    try {
      const data = JSON.parse(message);
      for (let id in players) {
        if (parseInt(id) !== playerId && players[id].readyState === WebSocket.OPEN) {
          players[id].send(JSON.stringify(data));
        }
      }
    } catch (e) {
      console.error("Failed to parse message:", e);
    }
  });

  ws.on("close", () => {
    console.log(`Player ${playerId} disconnected`);
    delete players[playerId];
    for (let id in players) {
      if (players[id].readyState === WebSocket.OPEN) {
        players[id].send(JSON.stringify({ type: "disconnect", playerid: playerId }));
      }
    }
  });
});

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
