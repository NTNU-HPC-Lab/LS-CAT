#include "includes.h"
__global__ void cudaDRectifier_backPropagate_kernel(double* x, double* dx, unsigned int size, double leakSlope, int shifting, double clipping)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
if (shifting > 0)
dx[i] /= (1 << shifting);
else if (shifting < 0)
dx[i] *= (1 << (-shifting));

if (clipping > 0.0) {
dx[i] *= (x[i] > clipping) ? 0.0 : (x[i] > 0.0)
? 1.0
: leakSlope;
}
else
dx[i] *= (x[i] > 0.0) ? 1.0 : leakSlope;
}
}