#include "includes.h"
__global__ void gpu_copy_velocity( const int num_atoms, const int offset, const int* g_group_contents, const double* g_vx_i, const double* g_vy_i, const double* g_vz_i, float* g_vx_o, float* g_vy_o, float* g_vz_o)
{
const int n = threadIdx.x + blockIdx.x * blockDim.x;
if (n < num_atoms) {
const int m = g_group_contents[offset + n];
g_vx_o[n] = g_vx_i[m];
g_vy_o[n] = g_vy_i[m];
g_vz_o[n] = g_vz_i[m];
}
}