#include "includes.h"
__global__ void computeTemporalSmoothRmatrices(const float* Rmatrices, uint32_t numSamples, uint32_t subArraySize, uint32_t numSubArrays, const uint32_t* subArraySizes, uint32_t temporalSmoothing, float* TempRmatrices)
{
int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
int sampleIdx = (blockIdx.y * gridDim.x) + blockIdx.x;

if (sampleIdx < numSamples)
{
int subArraySizeLocal = subArraySizes[sampleIdx];
int numelR = subArraySizeLocal*subArraySizeLocal;
int numelRfull = subArraySize*subArraySize;

int firstIdx = max(0, sampleIdx - (int)(temporalSmoothing));
int lastIdx = min((int)(numSamples)-1, sampleIdx + (int)(temporalSmoothing));

float scaling = 1.0f;
for (int matrixIdx = tIdx; matrixIdx < numelR; matrixIdx += blockDim.x*blockDim.y)
{
int colIdx = matrixIdx % subArraySizeLocal;
int rowIdx = matrixIdx / subArraySizeLocal;
int matrixStorageIdx = colIdx + rowIdx * subArraySize;

float finalEntry = 0.0f;
for (int tempIdx = firstIdx; tempIdx <= lastIdx; tempIdx++)
{
finalEntry += Rmatrices[matrixStorageIdx + tempIdx*numelRfull];
}
TempRmatrices[matrixStorageIdx + sampleIdx*numelRfull] = finalEntry*scaling;
}
}
}