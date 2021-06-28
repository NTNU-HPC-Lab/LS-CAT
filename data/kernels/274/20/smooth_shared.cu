#include "includes.h"
__global__ void smooth_shared(float * v_new, const float * v) {
extern __shared__ float s[];
int id = blockDim.x * blockIdx.x + threadIdx.x;
s[threadIdx.x + 1] = v[id];

if (threadIdx.x == 0) {
int start = blockDim.x * blockIdx.x;
int left = max(0, start - 1);
s[0] = v[left];
int end = blockDim.x * gridDim.x;
int right = min(end - 1, blockDim.x * blockIdx.x + blockDim.x);
s[blockDim.x + 1] = v[right];
}

__syncthreads();

int tid = threadIdx.x + 1;
v_new[id] = 0.25f * s[tid - 1] + 0.5f * s[tid] + 0.25f * s[tid + 1];
}