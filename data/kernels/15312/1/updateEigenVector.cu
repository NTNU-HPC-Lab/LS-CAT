#include "includes.h"
__global__ void updateEigenVector(float* d_b, float* d_temp, float* normAb, int n)
{
int index = threadIdx.x + blockDim.x * blockIdx.x;
int stride = 0;

while (index + stride < n) {
d_b[index] = d_temp[index] / *normAb;

stride += blockDim.x * gridDim.x;
}
}