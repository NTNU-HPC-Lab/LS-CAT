#include "includes.h"
__global__ void cuda_kernel_initRND(unsigned long seed, curandState *States)
{
int tid = threadIdx.x;
int bid = blockIdx.x;

int id    = bid*RND_BLOCK_SIZE + tid;
int pixel = bid*RND_BLOCK_SIZE + tid;

curand_init(seed, pixel, 0, &States[id]);
}