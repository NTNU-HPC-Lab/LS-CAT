#include "includes.h"
__global__ void update_presynaptic_activities_C_kernel(float* d_recent_presynaptic_activities_C, float* d_time_of_last_spike_to_reach_synapse, bool* d_stdp, float timestep, float current_time_in_seconds, size_t total_number_of_synapses, float synaptic_neurotransmitter_concentration_alpha_C, float decay_term_tau_C) {

int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
int idx = t_idx;
while (idx < total_number_of_synapses) {

if (d_stdp[idx] == true) {

float recent_presynaptic_activity_C = d_recent_presynaptic_activities_C[idx];

float new_recent_presynaptic_activity_C = (1 - (timestep/decay_term_tau_C)) * recent_presynaptic_activity_C;

if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {
new_recent_presynaptic_activity_C += timestep * synaptic_neurotransmitter_concentration_alpha_C * (1 - recent_presynaptic_activity_C);
}

if (recent_presynaptic_activity_C != new_recent_presynaptic_activity_C) {
d_recent_presynaptic_activities_C[idx] = new_recent_presynaptic_activity_C;
}

}

idx += blockDim.x * gridDim.x;

}

}