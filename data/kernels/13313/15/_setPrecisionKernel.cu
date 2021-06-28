#include "includes.h"
__global__ void _setPrecisionKernel(float* data, size_t size, int* precision)
{
unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
if (idx >= size) return;
int prec = precision[idx];
int mul = 1;
while(prec--) mul *= 10;
data[idx] = (float)(int)(data[idx]*mul);
data[idx] /= mul;
}