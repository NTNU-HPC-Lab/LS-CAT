#include "includes.h"
__global__ void rectified_linear_kernel( float4 * __restrict output, const float4 * __restrict input, float negative_slope, int elem_count)
{
int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
if (elem_id < elem_count)
{
float4 val = input[elem_id];
if (val.x < 0.0F)
val.x *= negative_slope;
if (val.y < 0.0F)
val.y *= negative_slope;
if (val.z < 0.0F)
val.z *= negative_slope;
if (val.w < 0.0F)
val.w *= negative_slope;
output[elem_id] = val;
}
}