#include "includes.h"
__global__ void izhikevich_update_membrane_potentials_kernel(float *d_membrane_potentials_v, float *d_states_u, float *d_param_a, float *d_param_b, float* d_current_injections, float* thresholds_for_action_potentials, float* last_spike_time_of_each_neuron, float* resting_potentials, float current_time_in_seconds, float timestep, size_t total_number_of_neurons) {

// We require the equation timestep in ms:
float eqtimestep = timestep*1000.0f;
// Get thread IDs
int idx = threadIdx.x + blockIdx.x * blockDim.x;
while (idx < total_number_of_neurons) {
// Update the neuron states according to the Izhikevich equations
float v_update = 0.04f*d_membrane_potentials_v[idx]*d_membrane_potentials_v[idx]
+ 5.0f*d_membrane_potentials_v[idx]
+ 140
- d_states_u[idx]
+ d_current_injections[idx];

d_membrane_potentials_v[idx] += eqtimestep*v_update;
d_states_u[idx] += eqtimestep*(d_param_a[idx] * (d_param_b[idx] * d_membrane_potentials_v[idx] -
d_states_u[idx]));

if (d_membrane_potentials_v[idx] >= thresholds_for_action_potentials[idx]){
d_membrane_potentials_v[idx] = resting_potentials[idx];
last_spike_time_of_each_neuron[idx] = current_time_in_seconds;
}

idx += blockDim.x * gridDim.x;
}
__syncthreads();
}