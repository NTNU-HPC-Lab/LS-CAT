#include "includes.h"
__global__ void __cumsumc(int nrows, int ncols, double *A, double *B) {
__shared__ double buff[32];
int i, j, k, lim;
double v, sum;
int icol = threadIdx.y + blockDim.y * blockIdx.x;
__syncthreads();
for (i = icol; i < ncols; i += blockDim.y * gridDim.x) {
sum = 0.0f;
for (j = 0; j < nrows; j += blockDim.x) {
v = 0;
if (j + threadIdx.x < nrows) {
v = A[j + threadIdx.x + i * nrows];
}
__syncthreads();
buff[threadIdx.x] = v;
lim = min(blockDim.x, nrows - j);
#pragma unroll
for (k = 1; k < lim; k = k + k) {
__syncthreads();
if (threadIdx.x >= k) {
v += buff[threadIdx.x - k];
}
__syncthreads();
buff[threadIdx.x] = v;
}
v += sum;
if (j + threadIdx.x < nrows) {
B[j + threadIdx.x + i * nrows] = v;
}
__syncthreads();
sum = buff[31];
__syncthreads();
}
}
}