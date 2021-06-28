#include "includes.h"
__global__ void computeTemporalSmoothRmatrices(const float* Rmatrices, uint32_t numSamples, uint32_t subArraySize, uint32_t numSubArrays, const uint32_t* subArraySizes, uint32_t temporalSmoothing, float* TempRmatrices)
{
int tIdx = (threadIdx.y * blockDim.x) + threadIdx.x;
int sampleIdx = blockIdx.x;
int scanlineIdxLocal = blockIdx.y;

if (sampleIdx < numSamples)
{
int subArraySizeLocal = subArraySizes[scanlineIdxLocal * numSamples + sampleIdx];
if (subArraySizeLocal > 0)
{
int numelR = subArraySizeLocal*(subArraySizeLocal + 1) /2;
int numelRfull = subArraySize*(subArraySize + 1) /2;

int firstIdx = max(0, sampleIdx - (int)(temporalSmoothing)) + scanlineIdxLocal * numSamples;
int lastIdx = min((int)(numSamples)-1, sampleIdx + (int)(temporalSmoothing)) + scanlineIdxLocal * numSamples;

float scaling = 1.0f;
for (int matrixIdx = tIdx; matrixIdx < numelR; matrixIdx += blockDim.x*blockDim.y)
{
float finalEntry = 0.0f;
for (int tempIdx = firstIdx; tempIdx <= lastIdx; tempIdx++)
{
finalEntry += Rmatrices[matrixIdx + tempIdx*numelRfull];
}
TempRmatrices[matrixIdx + (scanlineIdxLocal * numSamples + sampleIdx)*numelRfull] = finalEntry*scaling;
}
}
}
}