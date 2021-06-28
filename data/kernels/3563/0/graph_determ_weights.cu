#include "includes.h"
__global__ void graph_determ_weights(unsigned int* contact_mat_cum_row_indexes, unsigned int* contact_mat_column_indexes, float* contact_mat_values, unsigned int rows, unsigned int values, float* immunities, float* shedding_curve, unsigned int infection_length, float transmission_rate, int* infection_mat_values) {

unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;

curandState state;
curand_init(1234 + row, 0, 0, &state);
if (row < rows) {
for (int j=contact_mat_cum_row_indexes[row]; j<contact_mat_cum_row_indexes[row+1]; j++) {
float pinf_noshed = contact_mat_values[j] * transmission_rate * (1.0 - immunities[contact_mat_column_indexes[j]]);
int delay;
for (delay=1; delay<infection_length+1; delay++) {
//curand_uniform(&state)
if (curand_uniform(&state) < pinf_noshed * shedding_curve[delay - 1]) {
break;
}
}
if (delay > infection_length) {
delay = -1;
}
infection_mat_values[j] = delay;
}
}
}