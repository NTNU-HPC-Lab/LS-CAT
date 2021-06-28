#include "includes.h"

#define FLOAT_N 3214212.01

__global__ void calcdata(double* d_data, double* d_mean, int M, int N)
{
int j;
int i = blockDim.x * blockIdx.x + threadIdx.x+1;
if (i<=(N+1)) {
for (j = 1; j < (M+1); j++) {
d_data[i*(M+1) + j] -= d_mean[j];
}
}
}