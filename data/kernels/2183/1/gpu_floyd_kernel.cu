#include "includes.h"
__global__ void gpu_floyd_kernel(int k, int* adjacency_mtx, int* paths, int size)
{
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (col >= size)return;
int idx = size * blockIdx.y + col;

__shared__ int best;
if (threadIdx.x == 0)
best = adjacency_mtx[size * blockIdx.y + k];
__syncthreads();
if (best == INF)
return;
int tmp_b = adjacency_mtx[k * size + col];
if (tmp_b == INF)
return;
int cur = best + tmp_b;
if (cur < adjacency_mtx[idx]) {
adjacency_mtx[idx] = cur;
paths[idx] = k;
}
}