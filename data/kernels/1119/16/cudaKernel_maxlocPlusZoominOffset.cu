#include "includes.h"
__global__ void cudaKernel_maxlocPlusZoominOffset(float *offset, const int * padStart, const int * maxlocUpSample, const size_t nImages, float zoomInRatioX, float zoomInRatioY)
{
int imageIndex = threadIdx.x + blockDim.x *blockIdx.x; //image index
if (imageIndex < nImages)
{
int index=2*imageIndex;
offset[index] = padStart[index] + maxlocUpSample[index] * zoomInRatioX;
index++;
offset[index] = padStart[index] + maxlocUpSample[index] * zoomInRatioY;
}
}