#include "includes.h"
__global__ void rectified_linear_backprop_upd_kernel( float4 * __restrict input_errors, const float4 * __restrict output_errors, const uint4 * __restrict bits_buffer, float negative_slope, bool add_update_to_destination, int elem_count)
{
int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
if (elem_id < elem_count)
{
float4 val = output_errors[elem_id];
uint4 bits = bits_buffer[elem_id >> 5];
int lane_id = elem_id & 31;
unsigned int mask = (1 << lane_id);
if ((bits.x & mask) == 0)
val.x *= negative_slope;
if ((bits.y & mask) == 0)
val.y *= negative_slope;
if ((bits.z & mask) == 0)
val.z *= negative_slope;
if ((bits.w & mask) == 0)
val.w *= negative_slope;
if (add_update_to_destination)
{
float4 prv = input_errors[elem_id];
val.x += prv.x;
val.y += prv.y;
val.z += prv.z;
val.w += prv.w;
}
input_errors[elem_id] = val;
}
}