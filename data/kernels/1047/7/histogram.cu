#include "includes.h"
__global__ void histogram ( unsigned char *utime, unsigned int* histo, size_t n)
{
__shared__ unsigned int lhisto[512];
lhisto[threadIdx.x] = 0;
__syncthreads ();

int i = threadIdx.x + blockIdx.x*blockDim.x;
for (; i < n/2; i += blockDim.x*gridDim.x)
atomicAdd (lhisto+utime[i], 1);
for (; i < n; i += blockDim.x*gridDim.x)
atomicAdd ((lhisto+256)+utime[i], 1);
__syncthreads ();

// MUST run with 512 threads for this global accumulation to work
atomicAdd ( histo+threadIdx.x, lhisto[threadIdx.x]);
}