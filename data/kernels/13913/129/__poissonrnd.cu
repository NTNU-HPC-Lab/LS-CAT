#include "includes.h"
__global__ void __poissonrnd(int n, float *A, int *B, curandState *rstates) {
int id = threadIdx.x + blockDim.x * blockIdx.x;
int nthreads = blockDim.x * gridDim.x;
curandState rstate = rstates[id];
for (int i = id; i < n; i += nthreads) {
int cr = curand_poisson(&rstate, A[i]);
B[i] = cr;
}
}