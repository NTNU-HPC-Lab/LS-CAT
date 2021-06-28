#include "includes.h"
__global__ void weight_division_calc( float* sum_squared_afferent_values, float* afferent_weight_change_updater, float* weight_divisor, bool* neuron_in_plasticity_set, size_t total_number_of_neurons)
{
// Global Index
int idx = threadIdx.x + blockIdx.x * blockDim.x;

while (idx < total_number_of_neurons) {
if (neuron_in_plasticity_set[idx])
{
if ((sum_squared_afferent_values[idx] - afferent_weight_change_updater[idx] < 0.01))
printf("NORMALIZATION DIFF VERY LARGE. DANGER OF SYNAPSES ALL -> ZERO");
weight_divisor[idx] = sqrtf(sum_squared_afferent_values[idx] + afferent_weight_change_updater[idx]) / sqrtf(sum_squared_afferent_values[idx]);
}
idx += blockDim.x * gridDim.x;
}
__syncthreads();
}