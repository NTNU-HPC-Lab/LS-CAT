#include "includes.h"
__global__ void reduce_normal_eqs_64_GPU(float *d_C_reduced, float *d_C, int gridDim_x_normal_equations) {

int tid = threadIdx.x;
int bx = blockIdx.x;
// put data in shared memory

int ind = blockIdx.y * gridDim.x * gridDim_x_normal_equations * 64 +
bx * gridDim_x_normal_equations * 64 + tid;

__shared__ float DATA[64];

// load and sum the first 20 elements
float tmp = 0.0f;
for (int i = 0; i < gridDim_x_normal_equations; i++)
tmp += d_C[ind + i * 64];
DATA[tid] = tmp;

__syncthreads(); // ensure reading stage has finished

// reduction
if (tid < 32) { // warp-reduce
DATA[tid] += DATA[tid + 32];
__syncthreads();
DATA[tid] += DATA[tid + 16];
__syncthreads();
DATA[tid] += DATA[tid + 8];
__syncthreads();
DATA[tid] += DATA[tid + 4];
__syncthreads();
DATA[tid] += DATA[tid + 2];
__syncthreads();
DATA[tid] += DATA[tid + 1];
__syncthreads();
}

// write results
if (tid == 0)
d_C_reduced[blockIdx.y * gridDim.x + bx] = DATA[0];
}