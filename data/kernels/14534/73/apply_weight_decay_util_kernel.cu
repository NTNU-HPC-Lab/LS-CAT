#include "includes.h"
__global__ void apply_weight_decay_util_kernel( const float4 * __restrict learning_rates, float4 * __restrict weights, float weight_decay, int elem_count)
{
int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
if (elem_id < elem_count)
{
float4 val = learning_rates[elem_id];
float4 current_weight = weights[elem_id];
val.x = 1.0F - val.x * weight_decay;
val.y = 1.0F - val.y * weight_decay;
val.z = 1.0F - val.z * weight_decay;
val.w = 1.0F - val.w * weight_decay;
current_weight.x *= val.x;
current_weight.y *= val.y;
current_weight.z *= val.z;
current_weight.w *= val.w;
weights[elem_id] = current_weight;
}
}