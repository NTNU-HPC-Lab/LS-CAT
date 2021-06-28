#include "includes.h"
__global__ void ker_sparse_to_dense_assign(int n, const unsigned int *idx, float *src, float *trg) {
// Get our global thread ID
int id = blockIdx.x*blockDim.x+threadIdx.x;

// Make sure we do not go out of bounds
if (id < n)
trg[id] = src[idx[id]];
}