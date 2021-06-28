#include "includes.h"
__global__ void activate_array_leaky_kernel(float *x, int n)
{
int index = blockIdx.x*blockDim.x + threadIdx.x;
if (index < n) {
float val = x[index];
x[index] = (val > 0) ? val : val / 10;
}
}