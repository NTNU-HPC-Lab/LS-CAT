#include "includes.h"
__global__ void apply_gradient_with_weight_decay_util_kernel( const float2 * __restrict gradient, const float2 * __restrict learning_rates, float2 * __restrict weights, float weight_decay, int elem_count)
{
int elem_id = blockDim.x * blockIdx.x + threadIdx.x;
if (elem_id < elem_count)
{
float2 lr = learning_rates[elem_id];
float2 current_weight = weights[elem_id];
float2 grad = gradient[elem_id];
float2 new_weight;
new_weight.x = current_weight.x + lr.x * (grad.x - weight_decay * current_weight.x);
new_weight.y = current_weight.y + lr.y * (grad.y - weight_decay * current_weight.y);
weights[elem_id] = new_weight;
}
}