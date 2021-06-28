#include "includes.h"
__global__ void cudaDSaturation_backPropagate_kernel(double* x, double* dx, unsigned int size, double threshold)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
if (threshold != 0.0) {
dx[i] *= (x[i] > -threshold && x[i] < threshold)
? 1.0 : 0.0;
}
}
}