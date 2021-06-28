#include "includes.h"



__global__ void constant_add_kernel(const float *data_l, float constant, float *result, int total)
{
int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

if (idx / 2 < total) {
result[idx] = data_l[idx] + constant;
result[idx + 1] = data_l[idx + 1];
}
}