#include "includes.h"
__global__ void conductance_move_spikes_towards_synapses_kernel( int* d_spikes_travelling_to_synapse, float current_time_in_seconds, int* circular_spikenum_buffer, int* spikeid_buffer, int bufferloc, int buffersize, int total_number_of_synapses, float* d_time_of_last_spike_to_reach_synapse, int* postsynaptic_neuron_indices, float * neuron_wise_conductance_trace, int * synaptic_decay_id, int total_number_of_neurons, float * d_synaptic_efficacies_or_weights, float * d_biological_conductance_scaling_constants_lambda, float timestep){

int indx = threadIdx.x + blockIdx.x * blockDim.x;
while (indx < circular_spikenum_buffer[bufferloc]) {
int idx = spikeid_buffer[bufferloc*total_number_of_synapses + indx];

// Update Synapses
d_time_of_last_spike_to_reach_synapse[idx] = current_time_in_seconds;
int postsynaptic_neuron_id = postsynaptic_neuron_indices[idx];
int trace_id = synaptic_decay_id[idx];
float synaptic_efficacy = d_biological_conductance_scaling_constants_lambda[idx] * d_synaptic_efficacies_or_weights[idx];
atomicAdd(&neuron_wise_conductance_trace[total_number_of_neurons*trace_id + postsynaptic_neuron_id], synaptic_efficacy);

indx += blockDim.x * gridDim.x;
}
__syncthreads();
}