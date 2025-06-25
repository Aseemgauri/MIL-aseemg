"use strict";

process.stdout.write("🚀 Starting HumanFM server...\n");
console.log("Starting HumanFM server...");

const express = require("express");
console.log("Express loaded");

const net = require('net');
const fs = require('fs');
const path = require('path');
const app = express();
const multer = require("multer");
console.log("Multer loaded");

const sqlite3 = require("sqlite3");
const sqlite = require("sqlite");
console.log("SQLite loaded");

const USER_ERROR = 400;
const SERVER_ERROR = 500;
const PORT_NUM = 8000;

// Global variable to store the latest classification results (top 5)
let latestClassifications = [];
let lastClassificationTime = null;

// Define the 5 specific classes we care about (must match database)
const TARGET_CLASSES = ["Baby crying", "Cat", "Rooster", "Cricket", "Dog"];

console.log("Setting up Express middleware...");

app.use(express.urlencoded({extended: true}));
app.use(express.json());
app.use(multer().none());

console.log("Express middleware configured");

// // Temporary global variables to handle incoming C++ server data.
// let activeClasses = [];
// let soundLevels = [];

// // Create a new socket and connect to C++ server running on port 4000.
// const client = new net.Socket();
// client.connect(4000, '127.0.0.1', () => {
//     console.log('Connected to C++ server!');
// });

// // Handle incoming data from the server.
// client.on('data', (data) => {
//   let recClasses = [];
//   let recSounds = [];

//   // Read data related to active classes.
//   for (let i = 0; i < 20; i++) {
//     recClasses.push(data.readInt32LE(4 + i * 4));
//   }

//   // Read data related to sound levels for each class.
//   let offset = 4 + 20 * 4;
//   for (let i = 0; i < 20; i++) {
//     recSounds.push(data.readFloatLE(offset + i * 4));
//   }

//   activeClasses = recClasses;
//   soundLevels = recSounds;

//   console.log("Received activeClasses:", recClasses);
//   console.log("Received soundLevels:", recSounds);
// });

// // Temporary function to handle sending data to the server.
// function sendClientData() {
//   // Prepare buffer for sending updated data.
//   let buffer = Buffer.alloc(164);

//   // Writing our active classes to the buffer and introducing
//   // variable sound data.
//   for (let i = 0; i < 20; i++) {
//     buffer.writeInt32LE(activeClasses[i], 4 + i * 4);
//   }

//   // Only modifying the active classes.
//   let offset = 4 + 20 * 4;
//   for (let i = 0; i < 20; i++) {
//     if (activeClasses[i] == 1) {
//       let randFloat = Math.round(Math.random() * 100) / 100;
//       buffer.writeFloatLE(parseFloat(randFloat), offset + i * 4);
//     } else {
//       buffer.writeFloatLE(soundLevels[i], offset + i * 4);
//     }
//   }

//   // Simulate termination condition.
//   if (Math.random() < 0.1) {
//       buffer.writeInt32LE(1, 0);
//       console.log("Sending termination signal...");
//       return -1;
//   }

//   client.write(buffer);
//   return 0;
// }

// // Call the send function under a time interval.
// let intervalId = setInterval(() => {
//   console.log("Attempting to send server data.");
//   let termStatus = sendClientData();
//   if (termStatus == -1) {
//     clearInterval(intervalId);
//   }
// }, 10000);

// // Handle errors.
// client.on("error", (err) => {
//   console.error("Socket error:", err.message);
// });

// // Handle connection close.
// client.on('close', () => {
//     console.log('Connection closed.');
// });

// This function extracts data related to the classes from the database.
app.get("/HumanFM/classes", async function(req, res) {
  try {
    let classesTable = await getDBConnection();
    
    // Check if we have recent classification data (within last 5 seconds)
    const now = Date.now();
    const dataAge = lastClassificationTime ? now - lastClassificationTime : Infinity;
    const maxAge = 5000; // 5 seconds
    
    if (latestClassifications && latestClassifications.length > 0 && dataAge < maxAge) {
      console.log(`Serving ${latestClassifications.length} real classifications (${Math.round(dataAge/1000)}s old)`);
      
      // Get all the classes that are present (score >= 0.5)
      let presentClasses = latestClassifications.filter(cls => cls.score >= 0.5);
      
      if (presentClasses.length > 0) {
        // Return all present classes
        let allClasses = [];
        for (let cls of presentClasses) {
          console.log(`Looking up class in database: "${cls.label}"`);
          let classData = await classesTable.all(
            "SELECT * FROM classes WHERE name = ?",
            cls.label
          );
          if (classData.length > 0) {
            console.log(`Found class in database: ${classData[0].name} (ID: ${classData[0].id})`);
            allClasses.push(classData[0]);
          } else {
            console.log(`Class not found in database: "${cls.label}"`);
          }
        }
        await classesTable.close();
        console.log(`Returning ${allClasses.length} classes to frontend`);
        res.json(allClasses);
      } else {
        // No classes detected with high confidence
        console.log("No active classes in recent data");
        await classesTable.close();
        res.json([]);
      }
    } else {
      if (dataAge >= maxAge) {
        console.log(`Classification data too old (${Math.round(dataAge/1000)}s), returning empty`);
      } else {
        console.log("No real classifications available yet");
      }
      await classesTable.close();
      res.json([]);
    }
  } catch (err) {
    console.error("Error in /HumanFM/classes:", err);
    res.status(SERVER_ERROR).send("An error occurred on the server. Try again later");
  }
});

// This function updates the state of the database to account for changes in audio levels.
app.post("/HumanFM/classes/update", async function(req, res) {
  try {
    if (req.body) {
      let classesTable = await getDBConnection();
      let classId = req.body.id;
      let classLevel = req.body.level;

      await classesTable.run("UPDATE classes SET level = ? WHERE id = ?", [classLevel, classId]);
      await classesTable.close();
      res.type("text").send("success");
    } else {
      res.status(USER_ERROR).send("An error occurred extracting form data. Try again later");
    }
  } catch (err) {

  }
});

// This function presents the detected audio from the ML model based on the most
// recent live audio chuck from the user.
app.post("/HumanFM/classify", (req, res) => {
  try {
    if (req.body && req.body.classifications) {
      console.log("Received 5-class classification data:");
      latestClassifications = req.body.classifications;
      lastClassificationTime = Date.now(); // Store timestamp
      
      // Log which classes are present (score >= 0.5)
      let presentClasses = latestClassifications.filter(cls => cls.score >= 0.5);
      if (presentClasses.length > 0) {
        console.log("Active classes:", presentClasses.map(cls => cls.label).join(", "));
      } else {
        console.log("No active classes detected");
      }
      
      res.status(200).send("success");
    } else {
      res.status(400).send("error");
    }
  } catch (err) {
    console.error("Error in classify endpoint:", err);
    res.status(500).send("error");
  }
})

/**
 * Establishes a database connection to the database and returns the database object.
 * Any errors that occur should be caught in the function that calls this one.
 * @returns {Object} - The database object for the connection.
 */
async function getDBConnection() {
  const db = await sqlite.open({
    filename: 'humanfm.db',
    driver: sqlite3.Database
  });
  return db;
}

app.use(express.static('public'));
const PORT = process.env.PORT || PORT_NUM;
app.listen(PORT, '0.0.0.0', () => {
  process.stdout.write(`🌐 Server running on http://0.0.0.0:${PORT}\n`);
  console.log(`Server running on http://0.0.0.0:${PORT}`);
  console.log(`Access from other devices: http://<orange-pi-ip>:${PORT}`);
});

process.stdout.write("📡 Setting up Unix domain socket...\n");

// =============================================================================
// UNIX DOMAIN SOCKET SERVER FOR C++ CLIENT
// =============================================================================

const SOCKET_PATH = '/tmp/humanfm.sock';

// Remove existing socket file if it exists
if (fs.existsSync(SOCKET_PATH)) {
  fs.unlinkSync(SOCKET_PATH);
}

// Helper function to get class levels for the 5 target classes
async function getClassLevels() {
  try {
    const db = await getDBConnection();
    const levels = [];
    
    for (const className of TARGET_CLASSES) {
      const result = await db.all("SELECT level FROM classes WHERE name = ?", className);
      if (result.length > 0) {
        levels.push(parseFloat(result[0].level));
      } else {
        levels.push(0.0); // Default level if class not found
      }
    }
    
    await db.close();
    return levels;
  } catch (err) {
    console.error("Error getting class levels:", err);
    return [0.0, 0.0, 0.0, 0.0, 0.0]; // Default values on error
  }
}

// Helper function to get current class detection status (multihot vector)
function getClasses() {
  const now = Date.now();
  const dataAge = lastClassificationTime ? now - lastClassificationTime : Infinity;
  const maxAge = 5000; // 5 seconds
  
  // Initialize all classes as not detected
  const detectionVector = [0, 0, 0, 0, 0];
  
  if (latestClassifications && latestClassifications.length > 0 && dataAge < maxAge) {
    // Check each target class
    for (let i = 0; i < TARGET_CLASSES.length; i++) {
      const className = TARGET_CLASSES[i];
      const classification = latestClassifications.find(cls => cls.label === className);
      if (classification && classification.score >= 0.5) {
        detectionVector[i] = 1;
      }
    }
  }
  
  return detectionVector;
}

// Create Unix domain socket server
const socketServer = net.createServer((socket) => {
  console.log('C++ client connected to Unix socket');
  
  socket.on('data', async (data) => {
    try {
      const message = data.toString().trim();
      console.log(`Received socket command: ${message}`);
      
      let response;
      
      if (message === 'getClassLevels') {
        const levels = await getClassLevels();
        response = JSON.stringify({
          command: 'getClassLevels',
          data: levels,
          classes: TARGET_CLASSES
        });
        
      } else if (message === 'getClasses') {
        const detectionVector = getClasses();
        response = JSON.stringify({
          command: 'getClasses',
          data: detectionVector,
          classes: TARGET_CLASSES
        });
        
      } else {
        response = JSON.stringify({
          error: `Unknown command: ${message}`,
          availableCommands: ['getClassLevels', 'getClasses']
        });
      }
      
      socket.write(response + '\n');
      
    } catch (err) {
      console.error('Socket error:', err);
      const errorResponse = JSON.stringify({
        error: 'Internal server error'
      });
      socket.write(errorResponse + '\n');
    }
  });
  
  socket.on('end', () => {
    console.log('C++ client disconnected from Unix socket');
  });
  
  socket.on('error', (err) => {
    console.error('Socket connection error:', err);
  });
});

// Start Unix domain socket server
socketServer.listen(SOCKET_PATH, () => {
  process.stdout.write(`🔌 Unix domain socket server listening on ${SOCKET_PATH}\n`);
  console.log(`🔌 Unix domain socket server listening on ${SOCKET_PATH}`);
  console.log(`   Available commands: getClassLevels, getClasses`);
  console.log(`   Target classes: ${TARGET_CLASSES.join(', ')}`);
  process.stdout.write("✅ HumanFM server fully initialized!\n");
});

// Clean up socket on exit
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down servers...');
  if (fs.existsSync(SOCKET_PATH)) {
    fs.unlinkSync(SOCKET_PATH);
  }
  process.exit(0);
});

process.on('SIGTERM', () => {
  if (fs.existsSync(SOCKET_PATH)) {
    fs.unlinkSync(SOCKET_PATH);
  }
  process.exit(0);
});
