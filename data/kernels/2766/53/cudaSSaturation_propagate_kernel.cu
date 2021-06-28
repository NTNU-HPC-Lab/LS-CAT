#include "includes.h"
__global__ void cudaSSaturation_propagate_kernel(float* x, float* y, unsigned int size, int shifting, float threshold)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
float value = x[i];

if (shifting > 0)
value /= (1 << shifting);
else if (shifting < 0)
value *= (1 << (-shifting));

if (threshold != 0.0f) {
y[i] = (value < -threshold) ? -threshold
: (value > threshold) ? threshold
: value;
}
}
}