#include "includes.h"
__global__ void reset_states_u_after_spikes_kernel(float *d_states_u, float * d_param_d, float* d_last_spike_time_of_each_neuron, float current_time_in_seconds, size_t total_number_of_neurons) {

int idx = threadIdx.x + blockIdx.x * blockDim.x;
while (idx < total_number_of_neurons) {
if (d_last_spike_time_of_each_neuron[idx] == current_time_in_seconds) {

d_states_u[idx] += d_param_d[idx];

}
idx += blockDim.x * gridDim.x;
}
__syncthreads();
}