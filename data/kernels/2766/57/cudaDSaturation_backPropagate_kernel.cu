#include "includes.h"
__global__ void cudaDSaturation_backPropagate_kernel(double* x, double* dx, unsigned int size, int shifting, double threshold)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
if (shifting > 0)
dx[i] /= (1 << shifting);
else if (shifting < 0)
dx[i] *= (1 << (-shifting));

if (threshold != 0.0) {
dx[i] *= (x[i] > -threshold && x[i] < threshold)
? 1.0 : 0.0;
}
}
}