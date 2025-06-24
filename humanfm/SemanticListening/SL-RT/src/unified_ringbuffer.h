#ifndef UNIFIED_RINGBUFFER_H
#define UNIFIED_RINGBUFFER_H

#include <cstring>
#include <atomic>

// Minimal AudioIOBuffer implementation for audio_manager
class AudioIOBuffer {
private:
    static const size_t BUFFER_SIZE = 65536;  // 64KB buffer
    char buffer_[BUFFER_SIZE];
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};

public:
    AudioIOBuffer() = default;
    
    bool write(const char* data, size_t size) {
        size_t current_write = write_pos_.load();
        size_t current_read = read_pos_.load();
        
        // Calculate available space (circular buffer)
        size_t available_space;
        if (current_write >= current_read) {
            available_space = BUFFER_SIZE - (current_write - current_read) - 1;
        } else {
            available_space = current_read - current_write - 1;
        }
        
        if (size > available_space) {
            return false;  // Not enough space
        }
        
        // Write data (handle wrap-around)
        if (current_write + size <= BUFFER_SIZE) {
            std::memcpy(buffer_ + current_write, data, size);
        } else {
            size_t first_part = BUFFER_SIZE - current_write;
            std::memcpy(buffer_ + current_write, data, first_part);
            std::memcpy(buffer_, data + first_part, size - first_part);
        }
        
        write_pos_.store((current_write + size) % BUFFER_SIZE);
        return true;
    }
    
    bool read(char* data, size_t size) {
        size_t current_write = write_pos_.load();
        size_t current_read = read_pos_.load();
        
        // Calculate available data
        size_t available_data;
        if (current_write >= current_read) {
            available_data = current_write - current_read;
        } else {
            available_data = BUFFER_SIZE - (current_read - current_write);
        }
        
        if (size > available_data) {
            return false;  // Not enough data
        }
        
        // Read data (handle wrap-around)
        if (current_read + size <= BUFFER_SIZE) {
            std::memcpy(data, buffer_ + current_read, size);
        } else {
            size_t first_part = BUFFER_SIZE - current_read;
            std::memcpy(data, buffer_ + current_read, first_part);
            std::memcpy(data + first_part, buffer_, size - first_part);
        }
        
        read_pos_.store((current_read + size) % BUFFER_SIZE);
        return true;
    }
    
    size_t getAvailableBytes() const {
        size_t current_write = write_pos_.load();
        size_t current_read = read_pos_.load();
        
        if (current_write >= current_read) {
            return current_write - current_read;
        } else {
            return BUFFER_SIZE - (current_read - current_write);
        }
    }
    
    size_t getAvailableSpace() const {
        size_t current_write = write_pos_.load();
        size_t current_read = read_pos_.load();
        
        if (current_write >= current_read) {
            return BUFFER_SIZE - (current_write - current_read) - 1;
        } else {
            return current_read - current_write - 1;
        }
    }
};

#endif // UNIFIED_RINGBUFFER_H 