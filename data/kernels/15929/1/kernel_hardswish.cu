#include "includes.h"
__global__ void kernel_hardswish(const float *input_, float *output_, int n_data_size_)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
if (i >= n_data_size_)return;
if (input_[i] >= 3.0f)
{
output_[i] = input_[i];
}
else if (input_[i] <= -3.0f)
{
output_[i] = 0.0f;
}
else
{
output_[i] = input_[i] * (input_[i] + 3.0f) / 6.0f;
}
}