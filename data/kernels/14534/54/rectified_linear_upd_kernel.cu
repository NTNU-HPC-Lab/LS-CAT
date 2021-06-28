#include "includes.h"
__global__ void rectified_linear_upd_kernel( const float4 * __restrict input, float4 * __restrict output, uint4 * __restrict bits_buffer, float negative_slope, int elem_count)
{
int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
float4 val;
uint4 bits;
if (elem_id < elem_count)
val = input[elem_id];

#ifdef __CUDACC_VER_MAJOR__
#if __CUDACC_VER_MAJOR__ < 9
bits.x = __ballot(val.x < 0.0F ? 0 : 1);
bits.y = __ballot(val.y < 0.0F ? 0 : 1);
bits.z = __ballot(val.z < 0.0F ? 0 : 1);
bits.w = __ballot(val.w < 0.0F ? 0 : 1);
#else
bits.x = __ballot_sync(0xFFFFFFFF, val.x < 0.0F ? 0 : 1);
bits.y = __ballot_sync(0xFFFFFFFF, val.y < 0.0F ? 0 : 1);
bits.z = __ballot_sync(0xFFFFFFFF, val.z < 0.0F ? 0 : 1);
bits.w = __ballot_sync(0xFFFFFFFF, val.w < 0.0F ? 0 : 1);
#endif
#endif

if (elem_id < elem_count)
{
int lane_id = elem_id & 31;
if (lane_id == 0)
bits_buffer[elem_id >> 5] = bits;
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