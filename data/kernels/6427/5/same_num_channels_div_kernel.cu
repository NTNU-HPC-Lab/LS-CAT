#include "includes.h"



__global__ void same_num_channels_div_kernel(const float *data_l, const float *data_r, float *result, unsigned total)
{
int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

if (idx / 2 < total) {
result[idx] = (data_l[idx] * data_r[idx] + data_l[idx + 1] * data_r[idx + 1]) /
(data_r[idx] * data_r[idx] + data_r[idx + 1] * data_r[idx + 1]);
result[idx + 1] = (data_l[idx + 1] * data_r[idx] - data_l[idx] * data_r[idx + 1]) /
(data_r[idx] * data_r[idx] + data_r[idx + 1] * data_r[idx + 1]);
}
}