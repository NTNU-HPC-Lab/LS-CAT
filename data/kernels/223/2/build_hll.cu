#include "includes.h"
__global__ void build_hll(int n, unsigned int *in, unsigned int *out) {
int offset = (blockIdx.x * blockDim.x + threadIdx.x);
if (offset < n) {
// Extract the parts
unsigned int val = *(in + offset);
int bucket = val >> HLL_BUCKET_WIDTH;

// Update the maximum position
int pos = val & ((1 << HLL_BUCKET_WIDTH) - 1);

// Wait for all the maximums to be sync'd
atomicMax(&out[bucket], pos);
}
}