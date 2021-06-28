#include "includes.h"
__global__ void compute_l(double *dev_w, int n_patch)
{
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int N = n_patch * n_patch;
while (tid < N) {
dev_w[tid] = ((tid % (n_patch + 1) == 0) ? 1.0 : 0.0) - dev_w[tid];
tid += blockDim.x * gridDim.x;
}
}