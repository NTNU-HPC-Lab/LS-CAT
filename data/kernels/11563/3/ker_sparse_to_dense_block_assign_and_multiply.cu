#include "includes.h"
__global__ void ker_sparse_to_dense_block_assign_and_multiply(int n, const unsigned *idx, int bsize, float mult, float* src, float *trg) {
// Get our global thread ID
int id = blockIdx.x*blockDim.x+threadIdx.x;

// Make sure we do not go out of bounds
if (id < n*bsize)
trg[id] = src[idx[id/bsize]*bsize+id%bsize] * mult;
}