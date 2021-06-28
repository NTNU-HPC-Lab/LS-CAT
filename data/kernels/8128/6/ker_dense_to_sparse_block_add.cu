#include "includes.h"
__global__ void ker_dense_to_sparse_block_add(int n, const unsigned *idx, int bsize, float* src, float *trg) {
// Get our global thread ID
int id = blockIdx.x*blockDim.x+threadIdx.x;

// Make sure we do not go out of bounds
if (id < n*bsize)
atomicAdd(trg + idx[id/bsize]*bsize+id%bsize, src[id]);
}