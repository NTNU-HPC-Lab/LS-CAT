#include "includes.h"
__global__ void cudaDSaturation_propagate_kernel(double* x, double* y, unsigned int size, int shifting, double threshold)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
double value = x[i];

if (shifting > 0)
value /= (1 << shifting);
else if (shifting < 0)
value *= (1 << (-shifting));

if (threshold != 0.0) {
y[i] = (value < -threshold) ? -threshold
: (value > threshold) ? threshold
: value;
}
}
}