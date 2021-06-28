#include "includes.h"
__global__ void update_postsynaptic_activities_kernel( float timestep, size_t total_number_of_neurons, float * d_recent_postsynaptic_activities_D, float * d_last_spike_time_of_each_neuron, float current_time_in_seconds, float decay_term_tau_D, float model_parameter_alpha_D) {

int idx = threadIdx.x + blockIdx.x * blockDim.x;
while (idx < total_number_of_neurons) {

// if (d_stdp[idx] == 1) {

float recent_postsynaptic_activity_D = d_recent_postsynaptic_activities_D[idx];

float new_recent_postsynaptic_activity_D = (1 - (timestep/decay_term_tau_D)) * recent_postsynaptic_activity_D;

if (d_last_spike_time_of_each_neuron[idx] == current_time_in_seconds) {
new_recent_postsynaptic_activity_D += timestep * model_parameter_alpha_D * (1 - recent_postsynaptic_activity_D);
}

d_recent_postsynaptic_activities_D[idx] = new_recent_postsynaptic_activity_D;

// }

idx += blockDim.x * gridDim.x;

}
}