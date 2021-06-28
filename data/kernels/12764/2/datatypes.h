#ifndef DATATYPES_H
#define DATATYPES_H
#include <inttypes.h>

// Only stores the necessary information for further processing on gpu
typedef struct DVSEvent {
    // Use 32 Bit integer to use propper alignment on GPU
    uint32_t x;
    uint32_t y;
    //uint8_t polarity;
    uint32_t timestamp;
} DVSEvent;

#endif // DATATYPES_H
