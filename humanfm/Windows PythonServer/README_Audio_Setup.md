# Audio Device Configuration for HumanFM

## 🎤 **Audio Device Conflict Issues**

### **The Problem**
When JACK server is running for `realtime_processor`, it typically **reserves exclusive access** to the audio hardware. This means:

- ❌ GStreamer's `autoaudiosrc` may fail to access the microphone
- ❌ Other applications cannot directly access the audio device
- ❌ You might get "device busy" errors

### **The Solution**
Use **JACK-aware audio sources** that can connect to the existing JACK server instead of trying to access the hardware directly.

## 🔧 **Audio Backend Options**

### **1. JACK Audio Source (Recommended when JACK is running)**
```cpp
#define AUDIO_BACKEND_JACK 1      // Use jackaudiosrc
```
- ✅ Works with existing JACK server
- ✅ No device conflicts
- ✅ Professional audio routing
- ❌ Requires JACK to be running

### **2. PulseAudio Source**
```cpp
#define AUDIO_BACKEND_PULSE 1     // Use pulsesrc
```
- ✅ Works with PulseAudio
- ✅ User-friendly audio management
- ❌ May conflict with JACK

### **3. Direct ALSA Source**
```cpp
#define AUDIO_BACKEND_ALSA 1      // Use alsasrc
#define ALSA_DEVICE "hw:1,0"      // Specify exact device
```
- ✅ Direct hardware access
- ✅ Low latency
- ❌ Will conflict with JACK
- ❌ Requires knowing exact device

### **4. Automatic Source (Default)**
```cpp
#define AUDIO_BACKEND_AUTO 1      // Use autoaudiosrc
```
- ✅ Automatic detection
- ❌ **Will likely conflict with JACK**
- ❌ Unpredictable behavior

## 🔍 **Diagnosing Your Setup**

Run the diagnostic script to understand your audio configuration:

```bash
cd orange_pi/audio_processing
chmod +x audio_device_check.sh
./audio_device_check.sh
```

This will show you:
- Available ALSA devices
- PulseAudio status
- JACK server status
- Running audio processes
- GStreamer audio sources

## ⚙️ **Configuration Steps**

### **If JACK is Running (Recommended)**

1. **Check JACK status:**
   ```bash
   jack_control status
   jack_lsp  # List JACK ports
   ```

2. **Use JACK audio source:**
   Edit `chunks_improved.cpp`:
   ```cpp
   #define AUDIO_BACKEND_JACK 1
   #define AUDIO_BACKEND_AUTO 0
   ```

3. **Compile and test:**
   ```bash
   g++ -o chunks_jack chunks_improved.cpp `pkg-config --cflags --libs gstreamer-1.0`
   ./chunks_jack
   ```

### **If No JACK (Alternative)**

1. **Stop JACK if running:**
   ```bash
   jack_control stop
   ```

2. **Use ALSA or PulseAudio:**
   ```cpp
   #define AUDIO_BACKEND_ALSA 1    // or AUDIO_BACKEND_PULSE 1
   #define ALSA_DEVICE "hw:0,0"    // Check with aplay -l
   ```

## 🔗 **JACK Integration Benefits**

When using JACK audio source:

1. **No Device Conflicts**: Both `realtime_processor` and `chunks.cpp` use JACK
2. **Audio Routing**: Can route audio between applications
3. **Monitoring**: Can monitor what audio is being sent
4. **Professional Workflow**: Industry-standard audio server

### **JACK Connection Example**
```bash
# List available JACK ports
jack_lsp

# Connect GStreamer output to realtime_processor input (if needed)
jack_connect system:capture_1 gstreamer:input_1
```

## 🐛 **Troubleshooting**

### **"Device Busy" Error**
```
Error: Could not open audio device
```
**Solution**: JACK is likely running. Use `AUDIO_BACKEND_JACK`.

### **"jackaudiosrc not found"**
```
Error: no element "jackaudiosrc"
```
**Solution**: Install GStreamer JACK plugin:
```bash
sudo apt install gstreamer1.0-jack
```

### **No Audio Data**
1. Check microphone permissions
2. Verify JACK connections with `jack_lsp`
3. Test with: `gst-launch-1.0 jackaudiosrc ! fakesink`

### **JACK Server Not Running**
```bash
# Start JACK server
jack_control start

# Or start with specific settings
jackd -d alsa -r 48000 -p 256
```

## 📋 **Quick Reference**

| Scenario | Audio Backend | Configuration |
|----------|---------------|---------------|
| JACK running | `AUDIO_BACKEND_JACK` | Recommended |
| PulseAudio only | `AUDIO_BACKEND_PULSE` | Good |
| Direct ALSA | `AUDIO_BACKEND_ALSA` | Advanced |
| Don't know | Run diagnostic script | Start here |

## 🎯 **Recommended Workflow**

1. **Run diagnostic**: `./audio_device_check.sh`
2. **If JACK running**: Use `AUDIO_BACKEND_JACK`
3. **If no JACK**: Consider starting JACK for better integration
4. **Test with improved chunks**: Use `chunks_improved.cpp`
5. **Monitor connections**: Use `jack_lsp` and `jack_connect`

This setup ensures both your real-time audio processor and GStreamer client can work together without device conflicts! 