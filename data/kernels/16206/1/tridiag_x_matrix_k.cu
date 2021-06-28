#include "includes.h"
__global__ void tridiag_x_matrix_k(float p_d, float p_m, float p_u, float* u, int n)
{
// Identifies the thread working within a group
int tidx = threadIdx.x % n;
// Identifies the data concerned by the computations
int Qt = (threadIdx.x - tidx) / n;

extern __shared__ float sAds[];
float* su = (float*)&sAds[Qt * n];
su[threadIdx.x] = u[blockIdx.x * blockDim.x + threadIdx.x];
__syncthreads();

float temp;
if (tidx > 0 && tidx < n - 1)
temp = p_d * su[tidx - 1] + p_m * su[tidx] + p_u * su[tidx + 1];
else if (tidx == 0)
temp = p_m * su[tidx] + p_u * su[tidx + 1];
else
temp = p_d * su[tidx - 1] + p_m * su[tidx];

u[blockIdx.x * blockDim.x + threadIdx.x] = temp;
}