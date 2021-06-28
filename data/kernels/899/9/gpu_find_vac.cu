#include "includes.h"
__global__ void gpu_find_vac( const int num_atoms, const int correlation_step, const int num_correlation_steps, const float* g_vx, const float* g_vy, const float* g_vz, const float* g_vx_all, const float* g_vy_all, const float* g_vz_all, float* g_vac_x, float* g_vac_y, float* g_vac_z)
{
const int num_atoms_sq = num_atoms * num_atoms;
const int n1n2 = blockIdx.x * blockDim.x + threadIdx.x;
if (n1n2 >= num_atoms_sq)
return;
const int n1 = n1n2 / num_atoms;
const int n2 = n1n2 - n1 * num_atoms;
for (int k = 0; k < num_correlation_steps; ++k) {
int nc = correlation_step - k;
if (nc < 0)
nc += num_correlation_steps;
g_vac_x[nc * num_atoms_sq + n1n2] += g_vx[n1] * g_vx_all[k * num_atoms + n2];
g_vac_y[nc * num_atoms_sq + n1n2] += g_vy[n1] * g_vy_all[k * num_atoms + n2];
g_vac_z[nc * num_atoms_sq + n1n2] += g_vz[n1] * g_vz_all[k * num_atoms + n2];
}
}