#include "includes.h"
__global__ void set_with_value_util_kernel( float4 * __restrict buf, float v, int elem_count)
{
int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
if (elem_id < elem_count)
{
float4 val;
val.x = v;
val.y = v;
val.z = v;
val.w = v;
buf[elem_id] = val;
}
}