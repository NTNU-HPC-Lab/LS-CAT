#include "includes.h"
__global__ void gpu_find_vac( const int num_atoms, const int correlation_step, const double* g_mass, const double* g_vx, const double* g_vy, const double* g_vz, const double* g_vx_all, const double* g_vy_all, const double* g_vz_all, double* g_vac_x, double* g_vac_y, double* g_vac_z)
{
int tid = threadIdx.x;
int bid = blockIdx.x;
int size_sum = bid * num_atoms;
int number_of_rounds = (num_atoms - 1) / 128 + 1;
__shared__ double s_vac_x[128];
__shared__ double s_vac_y[128];
__shared__ double s_vac_z[128];
double vac_x = 0.0;
double vac_y = 0.0;
double vac_z = 0.0;

for (int round = 0; round < number_of_rounds; ++round) {
int n = tid + round * 128;
if (n < num_atoms) {
double mass = g_mass[n];
vac_x += mass * g_vx[n] * g_vx_all[size_sum + n];
vac_y += mass * g_vy[n] * g_vy_all[size_sum + n];
vac_z += mass * g_vz[n] * g_vz_all[size_sum + n];
}
}
s_vac_x[tid] = vac_x;
s_vac_y[tid] = vac_y;
s_vac_z[tid] = vac_z;
__syncthreads();

for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
if (tid < offset) {
s_vac_x[tid] += s_vac_x[tid + offset];
s_vac_y[tid] += s_vac_y[tid + offset];
s_vac_z[tid] += s_vac_z[tid + offset];
}
__syncthreads();
}

if (tid == 0) {
if (bid <= correlation_step) {
g_vac_x[correlation_step - bid] += s_vac_x[0];
g_vac_y[correlation_step - bid] += s_vac_y[0];
g_vac_z[correlation_step - bid] += s_vac_z[0];
} else {
g_vac_x[correlation_step + gridDim.x - bid] += s_vac_x[0];
g_vac_y[correlation_step + gridDim.x - bid] += s_vac_y[0];
g_vac_z[correlation_step + gridDim.x - bid] += s_vac_z[0];
}
}
}