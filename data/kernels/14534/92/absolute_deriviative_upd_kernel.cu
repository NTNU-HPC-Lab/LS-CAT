#include "includes.h"
__global__ void absolute_deriviative_upd_kernel( float4 * __restrict input_errors, const float4 * __restrict output_errors, const float4 * __restrict input_neurons, bool add_update_to_destination, int elem_count)
{
int elem_id = blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x) + threadIdx.x;
if (elem_id < elem_count)
{
float4 inp = input_neurons[elem_id];
float4 current_error = output_errors[elem_id];
if (inp.x < 0.0F)
current_error.x = -current_error.x;
if (inp.y < 0.0F)
current_error.y = -current_error.y;
if (inp.z < 0.0F)
current_error.z = -current_error.z;
if (inp.w < 0.0F)
current_error.w = -current_error.w;
float4 current_dst;
if (add_update_to_destination)
{
current_dst = input_errors[elem_id];
current_error.x += current_dst.x;
current_error.y += current_dst.y;
current_error.z += current_dst.z;
current_error.w += current_dst.w;
}
input_errors[elem_id] = current_error;
}
}