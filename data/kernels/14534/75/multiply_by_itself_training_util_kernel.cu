#include "includes.h"
__global__ void multiply_by_itself_training_util_kernel( const float4 * __restrict input_buf, float4 * __restrict output_buf, int elem_count)
{
int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
if (elem_id < elem_count)
{
float4 val = input_buf[elem_id];
val.x *= val.x;
val.y *= val.y;
val.z *= val.z;
val.w *= val.w;
output_buf[elem_id] = val;
}
}