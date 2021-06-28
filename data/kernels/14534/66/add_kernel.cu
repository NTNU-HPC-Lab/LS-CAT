#include "includes.h"
__global__ void add_kernel( float4 * __restrict output_buf, const float4 * __restrict input_buf1, const float4 * __restrict input_buf2, float alpha1, float alpha2, int elem_count)
{
int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
if (elem_id < elem_count)
{
float4 val1 = input_buf1[elem_id];
float4 val2 = input_buf2[elem_id];
val1.x = val1.x * alpha1 + val2.x * alpha2;
val1.y = val1.y * alpha1 + val2.y * alpha2;
val1.z = val1.z * alpha1 + val2.z * alpha2;
val1.w = val1.w * alpha1 + val2.w * alpha2;
output_buf[elem_id] = val1;
}
}