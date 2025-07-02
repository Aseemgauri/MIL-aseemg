"use strict";

console.log("🚀 Starting HumanFM server...");

const express = require("express");
console.log("Express loaded");

const net = require('net');
const fs = require('fs');
const app = express();
const multer = require("multer");
console.log("Multer loaded");

const sqlite3 = require("sqlite3");
const sqlite = require("sqlite");
console.log("SQLite loaded");

// Constants
const USER_ERROR = 400;
const SERVER_ERROR = 500;
const PORT_NUM = 8000;
const DATA_EXPIRY_TIME = 5000; // 5 seconds - how long to keep classification data
const DETECTION_THRESHOLD = 0.5; // Minimum score for class detection

// Global variable to store the latest classification results (top 5)
let latestClassifications = [];
let lastClassificationTime = null;

// Define the 5 specific classes we care about (must match Python server output)
const TARGET_CLASSES = ["Baby cry", "Cat", "Rooster", "Cricket", "Dog"];

// Hysteresis system: Track class states and activation timers
let previousClassStates = [0, 0, 0, 0, 0]; // Previous detection states
let currentDisplayStates = [0, 0, 0, 0, 0]; // What we're currently showing to clients
let classActivationTimers = [null, null, null, null, null]; // Timers for each class
const MINIMUM_ACTIVATION_TIME = 5000; // 5 seconds minimum display time

console.log("Setting up Express middleware...");

app.use(express.urlencoded({extended: true}));
app.use(express.json());
app.use(multer().none());

console.log("Express middleware configured");

// =============================================================================
// HTTP ENDPOINTS
// =============================================================================

// This function extracts data related to the classes from the database.
app.get("/HumanFM/classes", async function(req, res) {
  try {
    let classesTable = await getDBConnection();
    
    // Get hysteresis-processed class detection states (immediate activation, 5s minimum duration)
    const displayStates = getClasses();
    
    // Find which classes are currently being displayed (with hysteresis)
    let activeClassIndices = [];
    for (let i = 0; i < displayStates.length; i++) {
      if (displayStates[i] === 1) {
        activeClassIndices.push(i);
      }
    }
    
    if (activeClassIndices.length > 0) {
      console.log(`Serving ${activeClassIndices.length} hysteresis-processed classes: ${activeClassIndices.map(i => TARGET_CLASSES[i]).join(", ")}`);
      
      // Look up each active class in the database
      let allClasses = [];
      for (const classIndex of activeClassIndices) {
        const className = TARGET_CLASSES[classIndex];
        console.log(`Looking up class in database: "${className}"`);
        
        // First try exact match
        let classData = await classesTable.all(
          "SELECT * FROM classes WHERE name = ?",
          className
        );
        
        // If exact match fails, try partial matching
        if (classData.length === 0) {
          classData = await classesTable.all(
            "SELECT * FROM classes WHERE LOWER(name) LIKE '%' || LOWER(?) || '%' OR LOWER(?) LIKE '%' || LOWER(name) || '%'",
            [className, className]
          );
        }
        
        if (classData.length > 0) {
          console.log(`Found class in database: ${classData[0].name} (ID: ${classData[0].id})`);
          allClasses.push(classData[0]);
        } else {
          console.log(`Class not found in database: "${className}"`);
        }
      }
      
      await classesTable.close();
      console.log(`Returning ${allClasses.length} classes to frontend`);
      res.json(allClasses);
    } else {
      // No classes currently being displayed (after hysteresis processing)
      console.log("No active classes after hysteresis processing");
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
    console.error("Error updating class level:", err);
    res.status(SERVER_ERROR).send("An error occurred on the server. Try again later");
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
      
      // Log which classes are present (score >= threshold)
      let presentClasses = latestClassifications.filter(cls => cls.score >= DETECTION_THRESHOLD);
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
  console.log(`🌐 Server running on http://0.0.0.0:${PORT}`);
  console.log(`   Access from other devices: http://<orange-pi-ip>:${PORT}`);
});

console.log("📡 Setting up Unix domain socket...");

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
      // First try exact match
      let result = await db.all("SELECT level FROM classes WHERE name = ?", className);
      
      // If exact match fails, try partial matching
      if (result.length === 0) {
        result = await db.all(
          "SELECT level FROM classes WHERE LOWER(name) LIKE '%' || LOWER(?) || '%' OR LOWER(?) LIKE '%' || LOWER(name) || '%'",
          [className, className]
        );
      }
      
      if (result.length > 0) {
        levels.push(parseFloat(result[0].level));
      } else {
        levels.push(50.0); // Default level if class not found (50% volume)
      }
    }
    
    await db.close();
    return levels;
  } catch (err) {
    console.error("Error getting class levels:", err);
    return [50.0, 50.0, 50.0, 50.0, 50.0]; // Default values on error
  }
}

// Helper function to get current class detection status with hysteresis (multihot vector)
function getClasses() {
  const now = Date.now();
  const dataAge = lastClassificationTime ? now - lastClassificationTime : Infinity;
  
  // Get raw detections from latest classifications
  const rawDetectionVector = [0, 0, 0, 0, 0];
  
  if (latestClassifications && latestClassifications.length > 0 && dataAge < DATA_EXPIRY_TIME) {
    // Check each target class
    for (let i = 0; i < TARGET_CLASSES.length; i++) {
      const className = TARGET_CLASSES[i];
      
      // First try exact match
      let classification = latestClassifications.find(cls => cls.label === className);
      
      // If exact match fails, try partial matching (like Python server does)
      if (!classification) {
        const classNameLower = className.toLowerCase();
        classification = latestClassifications.find(cls => {
          const labelLower = cls.label.toLowerCase();
          // Check both directions: target in received OR received in target
          return classNameLower.includes(labelLower) || labelLower.includes(classNameLower);
        });
      }
      
      if (classification && classification.score >= DETECTION_THRESHOLD) {
        rawDetectionVector[i] = 1;
      }
    }
  }
  
  // Apply hysteresis logic for each class
  for (let i = 0; i < TARGET_CLASSES.length; i++) {
    const className = TARGET_CLASSES[i];
    const rawState = rawDetectionVector[i];
    const prevState = previousClassStates[i];
    const currentDisplayState = currentDisplayStates[i];
    
    if (rawState === 1) {
      // Class is currently detected
      if (prevState === 0) {
        // 0 → 1 transition: IMMEDIATE ACTIVATION
        console.log(`🟢 Class ACTIVATED: ${className} - showing immediately`);
        currentDisplayStates[i] = 1;
        classActivationTimers[i] = now; // Start timer
      } else {
        // 1 → 1 transition: staying active, keep timer unchanged
        currentDisplayStates[i] = 1;
      }
      
    } else {
      // Class is currently NOT detected (rawState === 0)
      if (currentDisplayState === 1) {
        // We're currently showing this class, check if we can deactivate it
        const activationTime = classActivationTimers[i];
        if (activationTime && (now - activationTime) >= MINIMUM_ACTIVATION_TIME) {
          // Class has been active for at least 5 seconds - allow deactivation
          console.log(`🔴 Class DEACTIVATED: ${className} - was active for ${Math.round((now - activationTime)/1000)}s`);
          currentDisplayStates[i] = 0;
          classActivationTimers[i] = null; // Reset timer
        } else {
          // Class has NOT been active for 5 seconds yet - keep showing it
          const timeRemaining = activationTime ? MINIMUM_ACTIVATION_TIME - (now - activationTime) : 0;
          console.log(`⏳ Class ${className} not detected but keeping active (${activationTime ? Math.round((now - activationTime)/1000) : 0}s/${Math.round(MINIMUM_ACTIVATION_TIME/1000)}s)`);
          currentDisplayStates[i] = 1; // Keep it active
        }
      } else {
        // Class not detected and not currently displayed - stay inactive
        currentDisplayStates[i] = 0;
      }
    }
  }
  
  // Update previous states for next cycle
  previousClassStates = [...rawDetectionVector];
  
  return currentDisplayStates;
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
  console.log(`🔌 Unix domain socket server listening on ${SOCKET_PATH}`);
  console.log(`   Available commands: getClassLevels, getClasses`);
  console.log(`   Target classes: ${TARGET_CLASSES.join(', ')}`);
  console.log("✅ HumanFM server fully initialized!");
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
