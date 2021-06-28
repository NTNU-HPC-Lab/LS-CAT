#include "includes.h"
__global__ void convolution_kernel_v1(float *d_output, float *d_input, float *d_filter, int num_row, int num_col, int filter_size)
{
int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
int idx_y = blockDim.y * blockIdx.y + threadIdx.y;

float result = 0.f;
for (int filter_row = -filter_size / 2; filter_row <= filter_size / 2; ++filter_row)
{
for (int filter_col = -filter_size / 2; filter_col <= filter_size / 2; ++filter_col)
{
// Find the global position to apply the given filter
int image_row = idx_y + filter_row;
int image_col = idx_x + filter_col;

float image_value = (image_row >= 0 && image_row < num_row && image_col >= 0 && image_col < num_col) ?
d_input[image_row * num_col + image_col] : 0.f;
float filter_value = d_filter[(filter_row + filter_size / 2) * filter_size + filter_col + filter_size / 2];

result += image_value * filter_value;
}
}

d_output[idx_y * num_col + idx_x] = result;
}