#include "includes.h"
__global__ void absolute_kernel( float4 * __restrict output, const float4 * __restrict input, int elem_count)
{
int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
if (elem_id < elem_count)
{
float4 val = input[elem_id];
val.x = fabsf(val.x);
val.y = fabsf(val.y);
val.z = fabsf(val.z);
val.w = fabsf(val.w);
output[elem_id] = val;
}
}