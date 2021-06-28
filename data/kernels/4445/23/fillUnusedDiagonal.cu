#include "includes.h"
__global__ void fillUnusedDiagonal(float* Rmatrices, uint32_t numSamples, uint32_t subArraySize, const uint32_t* subArraySizes)
{
int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
int sampleIdx = (blockIdx.y * gridDim.x) + blockIdx.x;

if (sampleIdx < numSamples)
{
int subArraySizeLocal = subArraySizes[sampleIdx];
int numelRfull = subArraySize * subArraySize;

if (subArraySize > subArraySizeLocal)
{
float* R = &Rmatrices[sampleIdx*numelRfull];
float diagEntry = R[subArraySize*subArraySize - 1];

for (int diagIdx = subArraySizeLocal + tIdx; diagIdx < subArraySize; diagIdx += blockDim.x*blockDim.y)
{
// subArraySize + 1 (instead of subArraySize) to follow the diagonal
int matrixIdx = diagIdx * (subArraySize + 1);

R[matrixIdx] = diagEntry;
}
}
}
}