#include "includes.h"
__global__ void __dds(int nrows, int nnz, double *A, double *B, int *Cir, int *Cic, double *P) {
__shared__ double parts[32*DDS_BLKY];
int jstart = ((long long)blockIdx.x) * nnz / gridDim.x;
int jend = ((long long)(blockIdx.x + 1)) * nnz / gridDim.x;
int tid = threadIdx.x + blockDim.x * threadIdx.y;
for (int j = jstart; j < jend ; j++) {
double sum = 0;
int aoff = nrows * Cir[j];
int boff = nrows * Cic[j];
for (int i = tid; i < nrows; i += blockDim.x * blockDim.y) {
sum += A[i + aoff] * B[i + boff];
}
parts[tid] = sum;
for (int i = 1; i < blockDim.x * blockDim.y; i *= 2) {
__syncthreads();
if (i + tid < blockDim.x * blockDim.y) {
parts[tid] = parts[tid] + parts[i + tid];
}
}
__syncthreads();
if (tid == 0) {
P[j] = parts[0];
}
__syncthreads();
}
}