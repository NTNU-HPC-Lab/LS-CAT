#include "includes.h"
__global__ void rnorm_all_in_one_kernel(float *vals, int n, float mu, float sigma)
{
// Usual block/thread indexing...
int myblock = blockIdx.x + blockIdx.y * gridDim.x;
int blocksize = blockDim.x * blockDim.y * blockDim.z;
int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
int idx = myblock * blocksize + subthread;

// Setup the RNG:
curandState rng_state;
curand_init(9131 + idx*17, 0, 0, &rng_state);

if (idx < n) {
vals[idx] = mu + sigma * curand_normal(&rng_state);
}
return;
}