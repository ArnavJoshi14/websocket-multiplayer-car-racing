const express = require("express");
const fs = require("fs");
const path = require("path");
const bodyParser = require("body-parser");
const WebSocket = require("ws");

const app = express();
const PORT = 3000;

// Middleware
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, "public")));

const http = require("http");
const server = http.createServer(app);

const wss = new WebSocket.Server({ server});

let players = {};

wss.on('connection', (ws) => {
    const playerId = Date.now(); // Simple unique ID
    players[playerId] = ws;
    console.log(`Player ${playerId} connected`);

    // ## FIX ADDED HERE: Send the new player their ID ##
    ws.send(JSON.stringify({ type: 'init', id: playerId }));

    ws.on('message', (message) => {
        // Parse the message to ensure it's valid before broadcasting
        try {
            const data = JSON.parse(message);
            // Broadcast to all other players
            for (let id in players) {
                if (parseInt(id) !== playerId && players[id].readyState === WebSocket.OPEN) {
                    players[id].send(JSON.stringify(data));
                }
            }
        } catch (e) {
            console.error("Failed to parse message or broadcast:", e);
        }
    });

    ws.on('close', () => {
        console.log(`Player ${playerId} disconnected`);
        delete players[playerId];
        // Optional: Broadcast a disconnect message so clients can remove the car
        for (let id in players) {
             if (players[id].readyState === WebSocket.OPEN) {
                 players[id].send(JSON.stringify({ type: 'disconnect', playerid: playerId }));
             }
        }
    });
});


// Start server
server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});


// // Ensure logs directory exists
// const logsDir = path.join(__dirname, "logs");
// if (!fs.existsSync(logsDir)) {
//     fs.mkdirSync(logsDir);
// }
// const logFile = path.join(logsDir, "game_logs.jsonl");

// // Endpoint to log data
// app.post("/log", (req, res) => {
//     const entry = req.body;

//     // Append JSONL (one JSON object per line)
//     fs.appendFile(logFile, JSON.stringify(entry) + "\n", (err) => {
//         if (err) {
//             console.error("Error writing log:", err);
//             return res.status(500).send("Error logging data");
//         }
//         res.sendStatus(200);
//     });
// });


