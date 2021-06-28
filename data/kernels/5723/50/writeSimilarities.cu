#include "includes.h"
__global__ void writeSimilarities(const float* nvccResults, int* activelayers, int writestep, int writenum, float* similarities, int active_slices, int slices)
{
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < active_slices)
{
float res = nvccResults[tid];
int slice = activelayers[tid];
for (int i = 0; i < writenum; ++i)
similarities[slices*writestep*i + slice] = res;
}
}