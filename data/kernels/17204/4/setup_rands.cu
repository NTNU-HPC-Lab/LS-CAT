#include "includes.h"
__global__ void setup_rands(curandState* rand, unsigned long seed, unsigned long N)
{

int x = threadIdx.x + (blockIdx.x * blockDim.x);
int y = threadIdx.y + (blockIdx.y * blockDim.y);
unsigned long tid = x + (y * blockDim.x * gridDim.x);

if(tid < N) curand_init(seed, tid, 0, &rand[tid]);

}