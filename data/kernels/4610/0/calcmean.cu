#include "includes.h"

#define FLOAT_N 3214212.01

__global__ void calcmean(double* d_data, double* d_mean, int M, int N)
{
int	i;
int j = blockDim.x * blockIdx.x + threadIdx.x+1;
if (j<=(M+1)) {
d_mean[j] = 0.0;
for (i = 1; i < (N+1); i++) {
d_mean[j] += d_data[i*(M+1) + j];
}
d_mean[j] /= FLOAT_N;
}
}