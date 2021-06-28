#include "includes.h"
__global__ void bit_reduce_count(const uint32_t *input_array, uint32_t *intBuf, uint32_t *countBuf, const uint16_t dataCount) {
extern __shared__ uint32_t sharedMem[];

const uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
const uint32_t a = input_array[i];
uint8_t size = 0;
if (a <= 0xff) {
size = 1;
} else if (a <= 0xffff) {
size = sizeof(uint16_t);
} else {
size = sizeof(uint32_t);
}

sharedMem[threadIdx.x] = size;

__syncthreads();

// really dumb addition
if (threadIdx.x == 1) {
uint32_t total = 0;
for (uint16_t i = 0; i < dataCount; i++) {
total += sharedMem[i];
sharedMem[i] = total;
}
countBuf[blockIdx.x] = total;
}
__syncthreads();

// block comapct
uint8_t* writeindex = (threadIdx.x + sharedMem[threadIdx.x] - size) + ((uint8_t*)&intBuf[(blockDim.x * blockIdx.x)]);
//uint8_t* writeindex = (threadIdx.x + sharedMem[threadIdx.x] - size) + ((uint8_t*)&intBuf[0]);

if (a <= 0xff) {
*writeindex = 1;
uint8_t b = static_cast<uint8_t>(a);
memcpy(writeindex+1, &b, sizeof(uint8_t));
} else if (a <= 0xffff) {
*writeindex = sizeof(uint16_t);
uint16_t s = static_cast<uint16_t>(a);
memcpy(writeindex+1, &s, sizeof(uint16_t));
} else {
*writeindex = sizeof(uint32_t);
memcpy(writeindex+1, &a, sizeof(uint32_t));
}

}