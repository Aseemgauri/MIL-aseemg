#ifndef UNIFIED_RINGBUFFER_H
#define UNIFIED_RINGBUFFER_H

#include <cstring>
#include <atomic>
#include <type_traits>
#include "audio_config.h"

// Lock-free ring buffer for structured data types
// Uses atomic operations for thread-safe single producer, single consumer operation
template<typename T, size_t Capacity>
class LockFreeRingBuffer {
private:
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    
    static constexpr size_t MASK = Capacity - 1;
    
    // Pre-allocated storage for elements
    alignas(64) T buffer_[Capacity];
    
    // Atomic indices for lock-free operation
    alignas(64) std::atomic<size_t> writeIndex_{0};
    alignas(64) std::atomic<size_t> readIndex_{0};

public:
    LockFreeRingBuffer() {
        // Initialize all elements to zero state (more efficient for POD types)
        if constexpr (std::is_trivially_constructible_v<T>) {
            std::memset(buffer_, 0, sizeof(buffer_));
        } else {
            // Fallback for non-trivial types
            for (size_t i = 0; i < Capacity; ++i) {
                new (&buffer_[i]) T();
            }
        }
    }
    
    ~LockFreeRingBuffer() {
        // Destroy all elements
        for (size_t i = 0; i < Capacity; ++i) {
            buffer_[i].~T();
        }
    }
    
    // Non-copyable, non-movable for safety
    LockFreeRingBuffer(const LockFreeRingBuffer&) = delete;
    LockFreeRingBuffer& operator=(const LockFreeRingBuffer&) = delete;
    LockFreeRingBuffer(LockFreeRingBuffer&&) = delete;
    LockFreeRingBuffer& operator=(LockFreeRingBuffer&&) = delete;
    
    // Try to push an element (non-blocking)
    // Returns true if successful, false if buffer is full
    bool tryPush(const T& item) {
        const size_t currentWrite = writeIndex_.load(std::memory_order_relaxed);
        const size_t nextWrite = (currentWrite + 1) & MASK;
        
        // Check if buffer is full
        if (nextWrite == readIndex_.load(std::memory_order_acquire)) {
            return false;
        }
        
        // Copy the item to the buffer
        buffer_[currentWrite] = item;
        
        // Update write index
        writeIndex_.store(nextWrite, std::memory_order_release);
        return true;
    }
    
    // Try to pop an element (non-blocking)
    // Returns true if successful, false if buffer is empty
    bool tryPop(T& item) {
        const size_t currentRead = readIndex_.load(std::memory_order_relaxed);
        
        // Check if buffer is empty
        if (currentRead == writeIndex_.load(std::memory_order_acquire)) {
            return false;
        }
        
        // Copy the item from the buffer
        item = buffer_[currentRead];
        
        // Update read index
        readIndex_.store((currentRead + 1) & MASK, std::memory_order_release);
        return true;
    }
    
    // Check if buffer is empty
    bool empty() const {
        return readIndex_.load(std::memory_order_acquire) == 
               writeIndex_.load(std::memory_order_acquire);
    }
    
    // Check if buffer is full
    bool full() const {
        const size_t currentWrite = writeIndex_.load(std::memory_order_acquire);
        const size_t nextWrite = (currentWrite + 1) & MASK;
        return nextWrite == readIndex_.load(std::memory_order_acquire);
    }
    
    // Get approximate size (may not be exact due to concurrent access)
    size_t size() const {
        const size_t write = writeIndex_.load(std::memory_order_acquire);
        const size_t read = readIndex_.load(std::memory_order_acquire);
        return (write - read) & MASK;
    }
    
    // Get capacity
    static constexpr size_t capacity() {
        return Capacity;
    }
    
    // Reset the buffer (not thread-safe, use only when no other threads are accessing)
    void reset() {
        writeIndex_.store(0, std::memory_order_release);
        readIndex_.store(0, std::memory_order_release);
    }
};

// Unified static ring buffer for both fixed-chunk and variable-size operations
// No dynamic allocation - all memory is pre-allocated based on audio configuration
template<size_t BufferCapacity>
class UnifiedRingBuffer {
private:
    // Pre-allocated buffer storage
    alignas(MEMORY_ALIGNMENT) char buffer_[BufferCapacity];
    
    // Atomic positions for thread-safe operation
    std::atomic<size_t> writePos_{0};
    std::atomic<size_t> readPos_{0};
    
    static constexpr size_t bufferSize_ = BufferCapacity;

public:
    UnifiedRingBuffer() {
        // Initialize buffer to zero
        std::memset(buffer_, 0, BufferCapacity);
    }
    
    // Non-copyable for safety
    UnifiedRingBuffer(const UnifiedRingBuffer&) = delete;
    UnifiedRingBuffer& operator=(const UnifiedRingBuffer&) = delete;
    
    // Variable-size write (for AudioManager compatibility)
    bool write(const char* data, size_t size) {
        if (!data || size == 0 || getAvailableSpace() < size) {
            return false;
        }
        
        const size_t currentWrite = writePos_.load(std::memory_order_relaxed);
        const size_t actualWritePos = currentWrite % bufferSize_;
        
        // Handle wrap-around case
        if (actualWritePos + size <= bufferSize_) {
            // No wrap-around needed
            std::memcpy(&buffer_[actualWritePos], data, size);
        } else {
            // Need to wrap around
            const size_t firstPart = bufferSize_ - actualWritePos;
            const size_t secondPart = size - firstPart;
            
            std::memcpy(&buffer_[actualWritePos], data, firstPart);
            std::memcpy(&buffer_[0], data + firstPart, secondPart);
        }
        
        // Update write position
        writePos_.store(currentWrite + size, std::memory_order_release);
        return true;
    }
    
    // Variable-size read (for AudioManager compatibility)
    bool read(char* output, size_t size) {
        if (!output || size == 0 || getAvailableBytes() < size) {
            return false;
        }
        
        const size_t currentRead = readPos_.load(std::memory_order_relaxed);
        const size_t actualReadPos = currentRead % bufferSize_;
        
        // Handle wrap-around case
        if (actualReadPos + size <= bufferSize_) {
            // No wrap-around needed
            std::memcpy(output, &buffer_[actualReadPos], size);
        } else {
            // Need to wrap around
            const size_t firstPart = bufferSize_ - actualReadPos;
            const size_t secondPart = size - firstPart;
            
            std::memcpy(output, &buffer_[actualReadPos], firstPart);
            std::memcpy(output + firstPart, &buffer_[0], secondPart);
        }
        
        // Update read position
        readPos_.store(currentRead + size, std::memory_order_release);
        return true;
    }
    
    // Fixed-size write (for StaticRingBuffer compatibility)
    void writeChunk(const char* data, size_t chunkSize) {
        if (!data || chunkSize == 0) return;
        
        const size_t currentWrite = writePos_.load(std::memory_order_relaxed);
        const size_t actualWritePos = currentWrite % bufferSize_;
        
        // Handle wrap-around case
        if (actualWritePos + chunkSize <= bufferSize_) {
            // No wrap-around needed
            std::memcpy(&buffer_[actualWritePos], data, chunkSize);
        } else {
            // Need to wrap around
            const size_t firstPart = bufferSize_ - actualWritePos;
            const size_t secondPart = chunkSize - firstPart;
            
            std::memcpy(&buffer_[actualWritePos], data, firstPart);
            std::memcpy(&buffer_[0], data + firstPart, secondPart);
        }
        
        // Update write position
        writePos_.store(currentWrite + chunkSize, std::memory_order_release);
    }
    
    // Get available bytes for reading
    size_t getAvailableBytes() const {
        const size_t write = writePos_.load(std::memory_order_acquire);
        const size_t read = readPos_.load(std::memory_order_acquire);
        
        if (write >= read) {
            return write - read;
        } else {
            // Handle wrap-around
            return bufferSize_ - read + write;
        }
    }
    
    // Get available space for writing
    size_t getAvailableSpace() const {
        return bufferSize_ - getAvailableBytes() - 1; // Leave one byte to distinguish full from empty
    }
    
    // Get the current write position
    size_t getWritePosition() const {
        return writePos_.load(std::memory_order_acquire) % bufferSize_;
    }
    
    // Get the current read position  
    size_t getReadPosition() const {
        return readPos_.load(std::memory_order_acquire) % bufferSize_;
    }
    
    // Reset the buffer (not thread-safe, use only during initialization)
    void reset() {
        writePos_.store(0, std::memory_order_release);
        readPos_.store(0, std::memory_order_release);
        std::memset(buffer_, 0, bufferSize_);
    }
    
    // Get buffer data for windowing operations (StaticRingBuffer compatibility)
    void getWindowData(char* output, size_t windowSize) const {
        if (!output || windowSize == 0) return;
        
        const size_t currentWrite = writePos_.load(std::memory_order_acquire);
        
        // Calculate start position for window (going backwards from write position)
        const size_t startPos = (currentWrite >= windowSize) ? 
                               (currentWrite - windowSize) % bufferSize_ :
                               (bufferSize_ + currentWrite - windowSize) % bufferSize_;
        
        // Copy window data with wrap-around handling
        if (startPos + windowSize <= bufferSize_) {
            // No wrap-around needed
            std::memcpy(output, &buffer_[startPos], windowSize);
        } else {
            // Need to wrap around
            const size_t firstPart = bufferSize_ - startPos;
            const size_t secondPart = windowSize - firstPart;
            
            std::memcpy(output, &buffer_[startPos], firstPart);
            std::memcpy(output + firstPart, &buffer_[0], secondPart);
        }
    }
    
    // Get buffer size
    static constexpr size_t getBufferSize() {
        return BufferCapacity;
    }
};

// Type aliases for specific use cases
using AudioIOBuffer = UnifiedRingBuffer<32768>;  // 32KB for audio I/O
using SlidingWindowBuffer = UnifiedRingBuffer<WINDOW_SIZE * sizeof(float) * 8>;  // For sliding window

#endif // UNIFIED_RINGBUFFER_H 