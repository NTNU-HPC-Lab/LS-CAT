#include "includes.h"
__global__ void cudaSquantize_kernel(float* x, float* y, unsigned int size, float minVal, float maxVal, unsigned int quantizationLevels, bool truncate)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

if (quantizationLevels > 1) {
const float scaling = (maxVal - minVal)
/ (float)(quantizationLevels - 1);

for (unsigned int i = index; i < size; i += stride) {
const float clamped = (x[i] < minVal) ? minVal :
(x[i] > maxVal) ? maxVal :
x[i];

if (truncate)
y[i] = (int)((clamped - minVal) / scaling) * scaling + minVal;
else {
y[i] = (int)round((clamped - minVal) / scaling)
* scaling + minVal;
}
}
}
else {
for (unsigned int i = index; i < size; i += stride)
y[i] = ((x[i] >= 0.0f) ? 1.0f : -1.0f);
}
}