#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "ringbuffer.h"

#define min(a, b) (a < b ? a : b)


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

bool rb_discard_bytes(ringbuffer_t *rb, uint32_t bytes_needed){
  rb->head = (rb->head + bytes_needed) % rb->size;
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
    amount = min(length, rb->size - rb->tail);
    
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
    amount = min(length, rb->size - cursor);
    
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
