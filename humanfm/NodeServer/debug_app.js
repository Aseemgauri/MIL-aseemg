console.log("Testing main app components...");

const express = require("express");
const net = require('net');
const fs = require('fs');
const path = require('path');
const app = express();
const multer = require("multer");
const sqlite3 = require("sqlite3");
const sqlite = require("sqlite");

console.log("✅ All modules loaded");

const USER_ERROR = 400;
const SERVER_ERROR = 500;
const PORT_NUM = 8000;

// Global variables
let latestClassifications = [];
let lastClassificationTime = null;
const TARGET_CLASSES = ["Speech", "Music", "Vehicle", "Animal", "Tools"];

console.log("✅ Variables initialized");

// Middleware
app.use(express.urlencoded({extended: true}));
app.use(express.json());
app.use(multer().none());

console.log("✅ Middleware added");

// Database connection function
async function getDBConnection() {
  const db = await sqlite.open({
    filename: 'humanfm.db',
    driver: sqlite3.Database
  });
  return db;
}

console.log("✅ Database function defined");

// Test database connection
console.log("Testing database connection...");
getDBConnection().then(async (db) => {
  console.log("✅ Database connection successful");
  await db.close();
}).catch((err) => {
  console.error("❌ Database connection failed:", err);
});

// Add routes one by one
app.get("/HumanFM/classes", async function(req, res) {
  console.log("Classes route called");
  res.json([]);
});

console.log("✅ Classes route added");

app.post("/HumanFM/classes/update", async function(req, res) {
  console.log("Update route called");
  res.send("success");
});

console.log("✅ Update route added");

app.post("/HumanFM/classify", (req, res) => {
  console.log("Classify route called");
  res.send("success");
});

console.log("✅ Classify route added");

// Check if public directory exists
console.log("Checking public directory...");
if (fs.existsSync('public')) {
  console.log("✅ Public directory exists");
  
  // List contents
  const files = fs.readdirSync('public');
  console.log("Public directory contents:", files);
  
  // Add static middleware
  app.use(express.static('public'));
  console.log("✅ Static middleware added");
} else {
  console.log("❌ Public directory not found");
}

// Start HTTP server
console.log("Starting HTTP server...");
const PORT = process.env.PORT || PORT_NUM;
app.listen(PORT, '0.0.0.0', () => {
  console.log(`✅ Server running on http://0.0.0.0:${PORT}`);
});

console.log("✅ Main app test completed"); 
