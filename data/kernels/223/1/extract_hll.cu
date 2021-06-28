#include "includes.h"
__global__ void extract_hll(int n, char *in, char *out) {
int offset = (blockIdx.x * blockDim.x + threadIdx.x);
if (offset < n) {
uint64_t *hash = (uint64_t*)(in + (HASH_WIDTH * offset));

// Get the first HLL_PREFIX_BITS to determine the bucket
int bucket = hash[0] >> (64 - HLL_PREFIX_BITS);

// Finds the position of the least significant 1 (0 to 64)
int position = __ffsll(hash[1]);

// Adjust for the limit of the bucket
if (position == 0) {
position = HLL_MAX_SCAN - 1;
} else
position = min(position, HLL_MAX_SCAN) - 1;

// Update the output
unsigned int *outp = ((unsigned int*)out) + offset;
*outp = ((bucket << HLL_BUCKET_WIDTH) | position);
}
}