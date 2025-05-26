#ifndef AUDIO_RINGBUFFER_H
#define AUDIO_RINGBUFFER_H

#include <vector>
#include <cstring>
#include <atomic>

class AudioRingBuffer {
public:
    AudioRingBuffer(size_t size) 
        : buffer(size, 0),
          writePos(0),
          readPos(0),
          bufferSize(size) {
    }

    // Write data to the buffer
    bool write(const char* data, size_t size) {
        if (getAvailableSpace() < size) {
            return false; // Not enough space
        }

        size_t writeIndex = writePos.load();
        
        // Handle wrap-around
        if (writeIndex + size <= bufferSize) {
            // No wrap-around needed
            std::memcpy(&buffer[writeIndex], data, size);
        } else {
            // Need to wrap around
            size_t firstPart = bufferSize - writeIndex;
            size_t secondPart = size - firstPart;
            
            std::memcpy(&buffer[writeIndex], data, firstPart);
            std::memcpy(&buffer[0], data + firstPart, secondPart);
        }
        
        writePos.store((writeIndex + size) % bufferSize);
        return true;
    }

    // Read data from the buffer
    bool read(char* output, size_t size) {
        if (getAvailableBytes() < size) {
            return false; // Not enough data
        }

        size_t readIndex = readPos.load();
        
        // Handle wrap-around
        if (readIndex + size <= bufferSize) {
            // No wrap-around needed
            std::memcpy(output, &buffer[readIndex], size);
        } else {
            // Need to wrap around
            size_t firstPart = bufferSize - readIndex;
            size_t secondPart = size - firstPart;
            
            std::memcpy(output, &buffer[readIndex], firstPart);
            std::memcpy(output + firstPart, &buffer[0], secondPart);
        }
        
        readPos.store((readIndex + size) % bufferSize);
        return true;
    }

    // Get available bytes to read
    size_t getAvailableBytes() const {
        size_t write = writePos.load();
        size_t read = readPos.load();
        
        if (write >= read) {
            return write - read;
        } else {
            return bufferSize - read + write;
        }
    }

    // Get available space to write
    size_t getAvailableSpace() const {
        return bufferSize - getAvailableBytes() - 1; // Leave one byte to distinguish full from empty
    }

    // Reset the buffer
    void reset() {
        writePos.store(0);
        readPos.store(0);
    }

private:
    std::vector<char> buffer;
    std::atomic<size_t> writePos;
    std::atomic<size_t> readPos;
    size_t bufferSize;
};

#endif // AUDIO_RINGBUFFER_H 