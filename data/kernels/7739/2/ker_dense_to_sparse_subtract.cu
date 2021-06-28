#include "includes.h"
__global__ void ker_dense_to_sparse_subtract(int n, const unsigned int *idx, float *src, float *trg) {
// Get our global thread ID
int id = blockIdx.x*blockDim.x+threadIdx.x;

// Make sure we do not go out of bounds
if (id < n)
atomicAdd(trg + idx[id], -src[id]);
}