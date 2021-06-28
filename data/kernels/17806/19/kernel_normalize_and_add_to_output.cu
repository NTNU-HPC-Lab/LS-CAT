#include "includes.h"
__global__ void kernel_normalize_and_add_to_output(float * dev_vol_in, float * dev_vol_out, float * dev_accumulate_weights, float * dev_accumulate_values)
{
unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
unsigned int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
unsigned int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

if (i >= c_volSize.x || j >= c_volSize.y || k >= c_volSize.z)
{
return;
}

// Index row major into the volume
long int out_idx = i + (j + k * c_volSize.y) * (c_volSize.x);

float eps = 1e-6;

// Divide the output volume's voxels by the accumulated splat weights
//   unless the accumulated splat weights are equal to zero
if (c_normalize)
{
if (abs(dev_accumulate_weights[out_idx]) > eps)
dev_vol_out[out_idx] = dev_vol_in[out_idx] + (dev_accumulate_values[out_idx] / dev_accumulate_weights[out_idx]);
else
dev_vol_out[out_idx] = dev_vol_in[out_idx];
}
else
dev_vol_out[out_idx] = dev_vol_in[out_idx] + dev_accumulate_values[out_idx];
}