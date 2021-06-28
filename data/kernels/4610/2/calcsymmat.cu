#include "includes.h"

#define FLOAT_N 3214212.01

__global__ void calcsymmat(double* d_data, double* d_symmat, int M, int N)
{
int	i, j2;
int j1 = blockDim.x * blockIdx.x + threadIdx.x+1;
if (j1<=(M+1)) {
for (j2 = j1; j2 < (M+1); j2++) {
d_symmat[j1*(M+1) + j2] = 0.0;
for (i = 1; i < N+1; i++) {
d_symmat[j1*(M+1) + j2] += d_data[i*(M+1) + j1] * d_data[i*(M+1) + j2];
}
d_symmat[j2*(M+1) + j1] = d_symmat[j1*(M+1) + j2];
}
}
}