#include "includes.h"
__global__ void gpu_calculate_force_square_max( const int size, const int number_of_rounds, const double* force_per_atom, double* force_square_max)
{
const int tid = threadIdx.x;

__shared__ double s_force_square[1024];
s_force_square[tid] = 0.0;

double force_square = 0.0;

for (int round = 0; round < number_of_rounds; ++round) {
const int n = tid + round * 1024;
if (n < size) {
const double f = force_per_atom[n];
if (f * f > force_square)
force_square = f * f;
}
}

s_force_square[tid] = force_square;
__syncthreads();

for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
if (tid < offset) {
if (s_force_square[tid + offset] > s_force_square[tid]) {
s_force_square[tid] = s_force_square[tid + offset];
}
}
__syncthreads();
}

if (tid == 0) {
force_square_max[0] = s_force_square[0];
}
}