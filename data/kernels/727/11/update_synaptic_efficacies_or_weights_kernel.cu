#include "includes.h"
__global__ void update_synaptic_efficacies_or_weights_kernel (float * d_recent_presynaptic_activities_C, float * d_recent_postsynaptic_activities_D, int* d_postsynaptic_neuron_indices, float* d_synaptic_efficacies_or_weights, float current_time_in_seconds, float * d_time_of_last_spike_to_reach_synapse, float * d_last_spike_time_of_each_neuron, float learning_rate_rho, int* d_plastic_synapse_indices, size_t total_number_of_plastic_synapses) {

int indx = threadIdx.x + blockIdx.x * blockDim.x;

while (indx < total_number_of_plastic_synapses) {
int idx = d_plastic_synapse_indices[indx];

float synaptic_efficacy_delta_g = d_synaptic_efficacies_or_weights[idx];
float new_synaptic_efficacy = synaptic_efficacy_delta_g;

float new_componet = 0.0;

int postsynaptic_neuron_index = d_postsynaptic_neuron_indices[idx];

if (d_last_spike_time_of_each_neuron[postsynaptic_neuron_index] == current_time_in_seconds) {
float recent_presynaptic_activity_C = d_recent_presynaptic_activities_C[idx];
float new_componet_addition = ((1 - synaptic_efficacy_delta_g) * recent_presynaptic_activity_C);
new_componet += new_componet_addition;
}

if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {
float recent_postsynaptic_activity_D = d_recent_postsynaptic_activities_D[postsynaptic_neuron_index];
new_componet -= (synaptic_efficacy_delta_g * recent_postsynaptic_activity_D);
}

if (new_componet != 0.0) {
new_componet = learning_rate_rho * new_componet;
new_synaptic_efficacy += new_componet;
}

if (synaptic_efficacy_delta_g != new_synaptic_efficacy) {
new_synaptic_efficacy = max(new_synaptic_efficacy, 0.0);
new_synaptic_efficacy = min(new_synaptic_efficacy, 1.0);

d_synaptic_efficacies_or_weights[idx] = new_synaptic_efficacy;
}



indx += blockDim.x * gridDim.x;
}
}