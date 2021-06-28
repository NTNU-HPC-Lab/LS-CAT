#include "includes.h"
__global__ void move(uint8_t *buf, uint32_t dest, uint32_t source, uint16_t bytesEach, const bool wipe) {
extern __shared__ uint8_t sharedMemT[];
const uint32_t i = threadIdx.x;

uint8_t *src = &buf[source];
for (uint16_t j = 0; j < bytesEach; j++)
{
sharedMemT[(i*bytesEach) + j] = src[(i*bytesEach) + j];
if (wipe){
src[(i*bytesEach) + j] = 0;
}
}

__syncthreads();

uint8_t *d = &buf[dest];
for (uint16_t j = 0; j < bytesEach; j++)
{
d[(i*bytesEach) + j] = sharedMemT[(i*bytesEach) + j];
}
}