#include "includes.h"
__global__ void update_positions( const int size, const double position_step, const double* force_per_atom, const double* position_per_atom, double* position_per_atom_temp)
{
const int n = blockIdx.x * blockDim.x + threadIdx.x;
if (n < size) {
const double position_change = force_per_atom[n] * position_step;
position_per_atom_temp[n] = position_per_atom[n] + position_change;
}
}