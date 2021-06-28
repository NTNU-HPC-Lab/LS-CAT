#include "includes.h"
__global__ void copy_buffer_util_kernel( const float4 * __restrict input_buf, float4 * __restrict output_buf, int elem_count)
{
int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
if (elem_id < elem_count)
output_buf[elem_id] = input_buf[elem_id];
}