#include "includes.h"
__global__ void THCudaTensor_copyUpperSymmetric(float *input, int n, int len)
{
for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < len; idx += 65535) {
const int r = idx % n;
const int c = idx / n;
if (r > c) {
input[idx] = input[r*n + c];
}
}
}