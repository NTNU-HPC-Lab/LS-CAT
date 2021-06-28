#include "includes.h"
__global__ void gpu_find_hac(const int Nc, const int Nd, const double* g_heat, double* g_hac)
{
//<<<Nc, 128>>>

__shared__ double s_hac_xi[128];
__shared__ double s_hac_xo[128];
__shared__ double s_hac_yi[128];
__shared__ double s_hac_yo[128];
__shared__ double s_hac_z[128];

int tid = threadIdx.x;
int bid = blockIdx.x;
int number_of_patches = (Nd - 1) / 128 + 1;
int number_of_data = Nd - bid;

s_hac_xi[tid] = 0.0;
s_hac_xo[tid] = 0.0;
s_hac_yi[tid] = 0.0;
s_hac_yo[tid] = 0.0;
s_hac_z[tid] = 0.0;

for (int patch = 0; patch < number_of_patches; ++patch) {
int index = tid + patch * 128;
if (index + bid < Nd) {
s_hac_xi[tid] += g_heat[index + Nd * 0] * g_heat[index + bid + Nd * 0] +
g_heat[index + Nd * 0] * g_heat[index + bid + Nd * 1];
s_hac_xo[tid] += g_heat[index + Nd * 1] * g_heat[index + bid + Nd * 1] +
g_heat[index + Nd * 1] * g_heat[index + bid + Nd * 0];
s_hac_yi[tid] += g_heat[index + Nd * 2] * g_heat[index + bid + Nd * 2] +
g_heat[index + Nd * 2] * g_heat[index + bid + Nd * 3];
s_hac_yo[tid] += g_heat[index + Nd * 3] * g_heat[index + bid + Nd * 3] +
g_heat[index + Nd * 3] * g_heat[index + bid + Nd * 2];
s_hac_z[tid] += g_heat[index + Nd * 4] * g_heat[index + bid + Nd * 4];
}
}
__syncthreads();

#pragma unroll
for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
if (tid < offset) {
s_hac_xi[tid] += s_hac_xi[tid + offset];
s_hac_xo[tid] += s_hac_xo[tid + offset];
s_hac_yi[tid] += s_hac_yi[tid + offset];
s_hac_yo[tid] += s_hac_yo[tid + offset];
s_hac_z[tid] += s_hac_z[tid + offset];
}
__syncthreads();
}

if (tid == 0) {
g_hac[bid + Nc * 0] = s_hac_xi[0] / number_of_data;
g_hac[bid + Nc * 1] = s_hac_xo[0] / number_of_data;
g_hac[bid + Nc * 2] = s_hac_yi[0] / number_of_data;
g_hac[bid + Nc * 3] = s_hac_yo[0] / number_of_data;
g_hac[bid + Nc * 4] = s_hac_z[0] / number_of_data;
}
}