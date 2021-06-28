#include "includes.h"
__global__ void weight_update( int* postsyn_neuron, bool* neuron_in_plasticity_set, float* current_weight, float* weight_divisor, int* d_plastic_synapse_indices, size_t total_number_of_plastic_synapses){

// Global Index
int indx = threadIdx.x + blockIdx.x * blockDim.x;

while (indx < total_number_of_plastic_synapses) {
int idx = d_plastic_synapse_indices[indx];
int postneuron = postsyn_neuron[idx];
if (neuron_in_plasticity_set[postneuron]){
float division_value = weight_divisor[postneuron];
//if (division_value != 1.0)
//printf("%f, %f, %f wat \n", division_value, current_weight[idx], (current_weight[idx] / division_value));
if (division_value != 1.0)
current_weight[idx] /= division_value;
}
indx += blockDim.x * gridDim.x;
}
}