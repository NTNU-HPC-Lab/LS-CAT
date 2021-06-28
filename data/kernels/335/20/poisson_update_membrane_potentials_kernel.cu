#include "includes.h"
__global__ void poisson_update_membrane_potentials_kernel(curandState_t* d_states, float *d_rates, float *d_membrane_potentials_v, float timestep, float * d_thresholds_for_action_potential_spikes, size_t total_number_of_input_neurons, int current_stimulus_index) {


int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
int idx = t_idx;
while (idx < total_number_of_input_neurons){

int rate_index = (total_number_of_input_neurons * current_stimulus_index) + idx;

float rate = d_rates[rate_index];

if (rate > 0.1) {

// Creates random float between 0 and 1 from uniform distribution
// d_states effectively provides a different seed for each thread
// curand_uniform produces different float every time you call it
float random_float = curand_uniform(&d_states[t_idx]);

// if the randomnumber is less than the rate
if (random_float < (rate * timestep)) {

// Puts membrane potential above default spiking threshold
d_membrane_potentials_v[idx] = d_thresholds_for_action_potential_spikes[idx] + 0.02;

}

}

idx += blockDim.x * gridDim.x;

}
__syncthreads();
}