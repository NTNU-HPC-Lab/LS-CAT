#include "includes.h"
__global__ void writeSimilarities(const float* nvccResults, int* activelayers, int writestep, int writenum, float* similarities, int active_patches, int patches)
{
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < active_patches)
{
float res = nvccResults[tid];
int patch = activelayers[tid];
for (int i = 0; i < writenum; ++i)
similarities[patches*writestep*i + patch] = res;
}
}