#ifndef _RING_BUFFER_H_
#define _RING_BUFFER_H_

#include "stdint.h"
#include "stdbool.h"

typedef struct{
  uint32_t size;
  uint8_t *buf;
  uint32_t head, tail;
  uint8_t val;
} ringbuffer_t;

// Allocates memory for a ringbuffer and initializes it
void rb_create(ringbuffer_t *rb, uint32_t size);

// Deallocate memory
void rb_destroy(ringbuffer_t *rb);

// Initializes a ringbugger from pre-allocated buffer
void rb_init(ringbuffer_t *rb, uint8_t *buf, uint32_t size);

// Discards bytes from front of ring buffer.
// USE WITH CAUTION: SERIALIZED STRUCTURES MAY BE BROKEN AFTER THIS
bool rb_discard_bytes(ringbuffer_t *rb, uint32_t bytes_needed);

bool rb_enough_space(ringbuffer_t *rb, uint32_t bytes_needed);
bool rb_put(ringbuffer_t *rb, const uint8_t *const data, uint32_t length);
bool rb_empty(ringbuffer_t *rb);
void rb_reset(ringbuffer_t *rb);

uint32_t rb_size(ringbuffer_t *rb);
uint8_t rb_get(ringbuffer_t *rb);

// WARNING: THESE DO NOT ADVANCE THE RING BUFFER
// IF THESE ARE USED, YOU MUST CALL rb_advance() TO FREE THE BYTES YOU JUST COPIED
bool rb_get_fast(ringbuffer_t *rb, uint8_t *buf, uint32_t length);
bool rb_get_fast_offset(ringbuffer_t *rb, uint8_t *buf, uint32_t length, uint32_t offset);

#define rb_at(rb, i) \
  (rb)->buf[((rb)->head + i)%(rb)->size]

#define rb_advance(rb, i) \
  (rb)->head = ((rb)->head + i) % (rb)->size




#endif
