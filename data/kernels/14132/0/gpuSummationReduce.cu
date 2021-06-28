#include "includes.h"


__global__ void gpuSummationReduce(float *in, float *out, int n)
{
extern __shared__ float sdata[];

// load shared mem
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

sdata[tid] = (i < n) ? in[i] : 0;

__syncthreads();

// do reduction in shared mem
for (unsigned int s=1; s < blockDim.x; s *= 2)
{
// modulo arithmetic is slow!
if ((tid % (2*s)) == 0)
{

sdata[tid] += sdata[tid + s]; //bigger number stored in low index
}

__syncthreads();
}

// write result for this block to global mem
if (tid == 0) out[blockIdx.x] = sdata[0];
}