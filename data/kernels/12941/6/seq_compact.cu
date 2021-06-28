#include "includes.h"
__global__ void seq_compact(uint8_t *intBuf, const uint16_t dataSize, uint32_t *sizeBuf) {
uint16_t writeIndex = 0;
for (uint16_t i = 0; i < dataSize; ++i) {
const uint16_t readIndex = i * 4;
uint8_t size = intBuf[readIndex];
memcpy(&intBuf[writeIndex], &intBuf[readIndex], size + 1);
writeIndex += size + 1;
}
sizeBuf[0] = writeIndex;

// zero out the rest of the buffer
const uint32_t int_buf_size = (dataSize * sizeof(uint32_t)) + (dataSize * sizeof(uint8_t));
memset(&intBuf[writeIndex], 0, int_buf_size - int_buf_size);
}