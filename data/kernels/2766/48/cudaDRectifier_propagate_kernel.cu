#include "includes.h"
__global__ void cudaDRectifier_propagate_kernel(double* x, double* y, unsigned int size, double leakSlope, int shifting, double clipping)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
double value = x[i];

if (shifting > 0)
value /= (1 << shifting);
else if (shifting < 0)
value *= (1 << (-shifting));

if (clipping > 0.0)
y[i] = (value > 0.0) ? min(value, clipping) : leakSlope * value;
else
y[i] = (value > 0.0) ? value : leakSlope * value;
}
}