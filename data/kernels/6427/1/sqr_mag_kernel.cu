#include "includes.h"



__global__ void sqr_mag_kernel(const float *data, float *result, int total)
{
int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

if (idx / 2 < total) {
result[idx] = data[idx] * data[idx] + data[idx + 1] * data[idx + 1];
result[idx + 1] = 0;
}
}