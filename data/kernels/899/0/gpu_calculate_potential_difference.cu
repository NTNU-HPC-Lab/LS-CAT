#include "includes.h"
__global__ void gpu_calculate_potential_difference( const int size, const int number_of_rounds, const double* potential_per_atom, const double* potential_per_atom_temp, double* potential_difference)
{
__shared__ double s_diff[1024];
s_diff[threadIdx.x] = 0.0;

double diff = 0.0f;

for (int round = 0; round < number_of_rounds; ++round) {
const int n = threadIdx.x + round * 1024;
if (n < size) {
diff += potential_per_atom_temp[n] - potential_per_atom[n];
}
}

s_diff[threadIdx.x] = diff;
__syncthreads();

for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
if (threadIdx.x < offset) {
s_diff[threadIdx.x] += s_diff[threadIdx.x + offset];
}
__syncthreads();
}

if (threadIdx.x == 0) {
potential_difference[0] = s_diff[0];
}
}