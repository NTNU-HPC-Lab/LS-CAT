#include "includes.h"
__device__ int GetVecIndex(int vecNumber, int dimCount, int *dimSizes, int measCount, int vecCount, int *dims)
{
unsigned long int index = 0;

for (int i = 0; i < dimCount; ++i)
index += (unsigned long int)dimSizes[i] * (unsigned long int)dims[i * vecCount + vecNumber];

return index;
}
__global__ void AddPackKernel(unsigned long int *codes, int *measures, int dimensionsCount, int *dimendionsSizes, int measuresCount, int currentCapacity, int fullCapacity, int packCount, int *packDimensions, int *packMeasures)
{
int currentVec = blockIdx.x * blockDim.x + threadIdx.x;

while (currentVec < packCount)
{
codes[currentCapacity + currentVec] = GetVecIndex(currentVec, dimensionsCount, dimendionsSizes, measuresCount, packCount, packDimensions);

for (int i = 0; i < measuresCount; ++i)
measures[i * fullCapacity + currentCapacity + currentVec] = packMeasures[i * packCount + currentVec];

currentVec += blockDim.x * gridDim.x;
}

}