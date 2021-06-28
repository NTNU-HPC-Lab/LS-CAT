#include "includes.h"
__global__ void all_dots(int n, int k, double* data_dots, double* centroid_dots, double* dots) {
__shared__ double local_data_dots[32];
__shared__ double local_centroid_dots[32];

int data_index = threadIdx.x + blockIdx.x * blockDim.x;
if ((data_index < n) && (threadIdx.y == 0)) {
local_data_dots[threadIdx.x] = data_dots[data_index];
}




int centroid_index = threadIdx.x + blockIdx.y * blockDim.y;
if ((centroid_index < k) && (threadIdx.y == 1)) {
local_centroid_dots[threadIdx.x] = centroid_dots[centroid_index];
}

__syncthreads();

centroid_index = threadIdx.y + blockIdx.y * blockDim.y;
if ((data_index < n) && (centroid_index < k)) {
dots[data_index + centroid_index * n] = local_data_dots[threadIdx.x] +
local_centroid_dots[threadIdx.y];
}
}