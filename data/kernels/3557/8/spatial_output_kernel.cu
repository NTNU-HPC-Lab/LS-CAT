#include "includes.h"
__global__ void spatial_output_kernel(unsigned int nbClass, unsigned int targetHeight, unsigned int targetWidth, float threshold, float* targetData, uint32_t* outputEstimated)
{
const int batchInputOffset = targetWidth * targetHeight * nbClass * blockIdx.z;
const int batchOutputOffset = targetWidth * targetHeight * blockIdx.z;

const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < targetWidth * targetHeight; i += stride)
{
unsigned int outputMax = 0;

if (nbClass > 1)
{
float maxVal = targetData[i + batchInputOffset];

for (unsigned int cls = 1; cls < nbClass; ++cls) {
const float tmp = targetData[i + cls*targetWidth*targetHeight
+ batchInputOffset];

if (tmp > maxVal) {
outputMax = cls;
maxVal = tmp;
}
}

outputEstimated[i + batchOutputOffset] = outputMax;
}
else if(nbClass == 1)
{
if(targetData[index] > threshold)
outputMax = 1;

const int estimatedLabel
= (targetData[i + batchInputOffset] > threshold);

outputEstimated[i + batchOutputOffset] = estimatedLabel;

}
}
}