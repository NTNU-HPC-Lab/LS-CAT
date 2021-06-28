#include "includes.h"
__global__ void global_reduction_kernel(float *data_out, float *data_in, int stride, int size)
{
int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

if (idx_x + stride < size) {
data_out[idx_x] += data_in[idx_x + stride];
}
}