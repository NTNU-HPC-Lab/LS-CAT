#include "includes.h"
__global__ void conductance_update_synaptic_conductances_kernel(float timestep, float * d_synaptic_conductances_g, float * d_synaptic_efficacies_or_weights, float * d_time_of_last_spike_to_reach_synapse, float * d_biological_conductance_scaling_constants_lambda, int total_number_of_synapses, float current_time_in_seconds, float * d_decay_terms_tau_g) {

int idx = threadIdx.x + blockIdx.x * blockDim.x;
while (idx < total_number_of_synapses) {

float synaptic_conductance_g = d_synaptic_conductances_g[idx];

float new_conductance = (1.0 - (timestep/d_decay_terms_tau_g[idx])) * synaptic_conductance_g;

if (d_time_of_last_spike_to_reach_synapse[idx] == current_time_in_seconds) {
float timestep_times_synaptic_efficacy = timestep * d_synaptic_efficacies_or_weights[idx];
float biological_conductance_scaling_constant_lambda = d_biological_conductance_scaling_constants_lambda[idx];
float timestep_times_synaptic_efficacy_times_scaling_constant = timestep_times_synaptic_efficacy * biological_conductance_scaling_constant_lambda;
new_conductance += timestep_times_synaptic_efficacy_times_scaling_constant;
}

if (synaptic_conductance_g != new_conductance) {
d_synaptic_conductances_g[idx] = new_conductance;
}

idx += blockDim.x * gridDim.x;
}
__syncthreads();

}