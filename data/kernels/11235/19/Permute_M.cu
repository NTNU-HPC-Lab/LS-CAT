#include "includes.h"






















__device__ curandState randomStates[256];



__global__ void Permute_M(const float* a, const float* permutedDimensions, const float* originalStrides, float* out, const float* permutedStrides, const int rank, const int n)
{
extern __shared__ float sdata[];

int blockId = blockIdx.y * gridDim.x + blockIdx.x;
int i = blockId * blockDim.x + threadIdx.x;

float* bufferIndices = &sdata[threadIdx.x * rank * 2];
float* resultIndices = &bufferIndices[rank];

if (i < n)
{
int flatIndex = i;

for (int y = 0; y < rank; y++)
{
bufferIndices[y] = (int) (flatIndex / originalStrides[y]);
flatIndex -= bufferIndices[y] * originalStrides[y];
}

for (int y = 0; y < rank; y++)
{
resultIndices[y] = bufferIndices[(int) permutedDimensions[y]];
}

int permutedIndex = 0;

for (int y = 0; y < rank; y++)
{
permutedIndex += resultIndices[y] * permutedStrides[y];
}

out[permutedIndex] = a[i];
}
}