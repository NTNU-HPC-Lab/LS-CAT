#include "includes.h"
__global__ void rpois_kernel(curandState *state, int *vals, int n, double lambda)
{
// Usual block/thread indexing...
int myblock = blockIdx.x + blockIdx.y * gridDim.x;
int blocksize = blockDim.x * blockDim.y * blockDim.z;
int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
int idx = myblock * blocksize + subthread;

if (idx < n) {
vals[idx] = curand_poisson(&state[idx], lambda);
}
return;
}