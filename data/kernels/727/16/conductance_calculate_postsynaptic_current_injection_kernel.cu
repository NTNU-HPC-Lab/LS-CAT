#include "includes.h"
__global__ void conductance_calculate_postsynaptic_current_injection_kernel( float* decay_term_values, float* reversal_values, int num_decay_terms, int* synapse_decay_values, float* neuron_wise_conductance_traces, float* d_neurons_current_injections, float * d_membrane_potentials_v, float timestep, size_t total_number_of_neurons){

int idx = threadIdx.x + blockIdx.x * blockDim.x;
while (idx < total_number_of_neurons) {

float membrane_potential_v = d_membrane_potentials_v[idx];

for (int decay_id = 0; decay_id < num_decay_terms; decay_id++){
if (decay_id == 0)
d_neurons_current_injections[idx] = 0.0f;
float synaptic_conductance_g = neuron_wise_conductance_traces[idx + decay_id*total_number_of_neurons];
// First decay the conductance values as required
synaptic_conductance_g *= expf(- timestep / decay_term_values[decay_id]);
neuron_wise_conductance_traces[idx + decay_id*total_number_of_neurons] = synaptic_conductance_g;
d_neurons_current_injections[idx] += synaptic_conductance_g * (reversal_values[decay_id] - membrane_potential_v);
}

idx += blockDim.x * gridDim.x;

}
}