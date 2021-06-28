#include "includes.h"
__global__ void rexpo_kernel(curandState *state, float *vals, int n, float lambda)
{
// Usual block/thread indexing...
int myblock = blockIdx.x + blockIdx.y * gridDim.x;
int blocksize = blockDim.x * blockDim.y * blockDim.z;
int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
int idx = myblock * blocksize + subthread;

if (idx < n){
vals[idx] = -log(curand_uniform(&state[idx]))/lambda;
}
return;
}