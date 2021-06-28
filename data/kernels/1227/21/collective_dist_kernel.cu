#include "includes.h"
__device__ double get_collective_dist(int *dist, int rows, int cols, int col) {
double sum = 0;
for (int i = 0; i < rows; i++) {
if (dist[i * cols + col] == 0) {
return 0;
}
sum += (1 / (double)dist[i * cols + col]);
}
return sum;
}
__global__ void collective_dist_kernel(int *dist, int rows, int cols, double *col_dist)
{
int tid = threadIdx.x + blockIdx.x * blockDim.x;
while (tid < cols) {
col_dist[tid] = get_collective_dist(dist, rows, cols, tid);
tid += blockDim.x * gridDim.x;
}
}