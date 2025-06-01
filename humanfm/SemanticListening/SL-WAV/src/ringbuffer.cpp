#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstring>
#include <stdexcept>
#include <algorithm>

#include "ringbuffer.h"


void rb_create(ringbuffer_t *rb, uint32_t size){
  uint8_t *buf = new uint8_t[size];
  rb_init(rb, buf, size);
}

void rb_destroy(ringbuffer_t *rb){
  if(rb->buf){
    free(rb->buf);
  }
  rb_init(rb, NULL, 0);
}

void rb_init(ringbuffer_t *rb, uint8_t *buf, uint32_t size){
  rb->size = size;
  rb->buf = buf;
  rb->head = rb->tail = 0;

  if(rb->buf){
    rb->val = rb->buf[rb->head];
  }
}

uint32_t rb_size(ringbuffer_t *rb){
  if(rb->size == 0) return 0;

  return (rb->size - rb->head + rb->tail) % rb->size;
}

bool rb_empty(ringbuffer_t *rb){
  return rb->head == rb->tail;
}

bool rb_discard_bytes(ringbuffer_t *rb, uint32_t bytes) {
    if (bytes > rb->size) {
        return false;
    }
    rb->head = (rb->head + bytes) % rb->size;
    rb->size -= bytes;
    return true;
}

bool rb_enough_space(ringbuffer_t *rb, uint32_t bytes_needed){
  if(rb->size == 0) return 0;

  return rb_size(rb) + bytes_needed < rb->size - 1;
}

bool rb_put(ringbuffer_t *rb, const uint8_t * const data, uint32_t length){
  if(rb_size(rb) + length >= rb->size - 1) return 0;

  const uint8_t *ptr = data;
  uint32_t amount;
  while(length > 0){
    amount = std::min(length, rb->size - rb->tail);
    
    memcpy(rb->buf + rb->tail, ptr, amount);
  
    rb->tail += amount;
  
    if(rb->tail == rb->size){
      rb->tail = 0;
    }

    ptr += amount;
    length -= amount;
  }
  return 1;
}

uint8_t rb_get(ringbuffer_t *rb){
  if(rb_empty(rb)) return 0;
  
  rb->val = rb->buf[rb->head];
  
  rb->head++;
  if(rb->head >= rb->size) rb->head -= rb->size;
  
  return rb->val;
}

bool rb_get_fast(ringbuffer_t *rb, uint8_t *buf, uint32_t length){
  return rb_get_fast_offset(rb, buf, length, 0);
}

bool rb_get_fast_offset(ringbuffer_t *rb, uint8_t *buf, uint32_t length, uint32_t offset){
  if(rb_size(rb) < length + offset) return 0;
  
  uint32_t amount;
  uint32_t cursor = (rb->head + offset) % rb->size;
  while(length > 0){
    amount = std::min(length, rb->size - cursor);
    
    memcpy(buf, rb->buf + cursor, amount);
  
    cursor += amount;
    buf += amount;

    if(cursor == rb->size){
      cursor = 0;
    }

    length -= amount;
  }

  return 1;
}

void rb_reset(ringbuffer_t *rb){
  rb_init(rb, rb->buf, rb->size);
}

// Fixed-size ring buffer class implementation
RingBuffer::RingBuffer(size_t chunkSize) 
    : CHUNK_SIZE(chunkSize),
      BUFFER_SIZE(3 * chunkSize),
      writePos(0),
      readPos(0) {
    // Validate buffer size
    if (BUFFER_SIZE > sizeof(buffer)) {
        throw std::runtime_error("Buffer size exceeds static allocation");
    }
    memset(buffer, 0, BUFFER_SIZE);
}

// Copy constructor
RingBuffer::RingBuffer(const RingBuffer& other)
    : CHUNK_SIZE(other.CHUNK_SIZE),
      BUFFER_SIZE(other.BUFFER_SIZE),
      writePos(other.writePos),
      readPos(other.readPos) {
    memcpy(buffer, other.buffer, BUFFER_SIZE);
}

// Assignment operator
RingBuffer& RingBuffer::operator=(const RingBuffer& other) {
    if (this != &other) {
        const_cast<size_t&>(CHUNK_SIZE) = other.CHUNK_SIZE;
        const_cast<size_t&>(BUFFER_SIZE) = other.BUFFER_SIZE;
        memcpy(buffer, other.buffer, BUFFER_SIZE);
        writePos = other.writePos;
        readPos = other.readPos;
    }
    return *this;
}

// Add a chunk of data to the buffer
void RingBuffer::write(const char* data) {
    // Write the new chunk
    std::memcpy(&buffer[writePos], data, CHUNK_SIZE);
    
    // Update write position
    writePos = (writePos + CHUNK_SIZE) % BUFFER_SIZE;
}

// Read a chunk of data from the buffer
void RingBuffer::read(char* output) {
    // Read the chunk
    std::memcpy(output, &buffer[readPos], CHUNK_SIZE);
    
    // Update read position
    readPos = (readPos + CHUNK_SIZE) % BUFFER_SIZE;
}

// Get the current state of the buffer (return wrapper for compatibility)
RingBuffer::BufferWrapper RingBuffer::getBuffer() const {
    return {buffer, BUFFER_SIZE};
}

// Get the current write position
size_t RingBuffer::getWritePosition() const {
    return writePos;
}

// Get the current read position
size_t RingBuffer::getReadPosition() const {
    return readPos;
}
