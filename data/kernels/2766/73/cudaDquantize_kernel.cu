#include "includes.h"
__global__ void cudaDquantize_kernel(double* x, double* y, unsigned int size, double minVal, double maxVal, unsigned int quantizationLevels, bool truncate)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

if (quantizationLevels > 1) {
const double scaling = (maxVal - minVal)
/ (double)(quantizationLevels - 1);

for (unsigned int i = index; i < size; i += stride) {
const double clamped = (x[i] < minVal) ? minVal :
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
y[i] = ((x[i] >= 0.0) ? 1.0 : -1.0);
}
}