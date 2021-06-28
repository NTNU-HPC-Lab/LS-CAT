#include "includes.h"
__global__ void SumCentroids(float* delta, float* sumDelta, int numOfCentroids, int numOfElements)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;

if (id < numOfCentroids * NUM_SUMS)
{
float sum = 0;

for (int i = 0; i < numOfElements; i++)
{
sum += delta[numOfElements * id + i];
}

sumDelta[id] = sum;
}
}