#include "includes.h"
__global__ void gpu_copy_mass( const int num_atoms, const int* g_group_contents, const double* g_mass_i, double* g_mass_o)
{
const int n = threadIdx.x + blockIdx.x * blockDim.x;
if (n < num_atoms) {
g_mass_o[n] = g_mass_i[g_group_contents[n]];
}
}