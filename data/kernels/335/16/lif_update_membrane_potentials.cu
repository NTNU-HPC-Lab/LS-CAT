#include "includes.h"
__global__ void lif_update_membrane_potentials(float *d_membrane_potentials_v, float * d_membrane_resistances_R, float * d_membrane_time_constants_tau_m, float * d_resting_potentials, float* d_current_injections, float timestep, size_t total_number_of_neurons){


// // Get thread IDs
int idx = threadIdx.x + blockIdx.x * blockDim.x;
while (idx < total_number_of_neurons) {

float equation_constant = timestep / d_membrane_time_constants_tau_m[idx];
float membrane_potential_Vi = d_membrane_potentials_v[idx];
float current_injection_Ii = d_current_injections[idx];
float resting_potential_V0 = d_resting_potentials[idx];
float temp_membrane_resistance_R = d_membrane_resistances_R[idx];

float new_membrane_potential = equation_constant * (resting_potential_V0 + temp_membrane_resistance_R * current_injection_Ii) + (1 - equation_constant) * membrane_potential_Vi;

d_membrane_potentials_v[idx] = new_membrane_potential;

idx += blockDim.x * gridDim.x;

}
__syncthreads();
}