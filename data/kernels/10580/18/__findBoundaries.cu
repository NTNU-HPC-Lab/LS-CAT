#include "includes.h"
__global__ void __findBoundaries(long long *keys, int *jc, int n, int njc, int shift) {
__shared__ int dbuff[1024];
int i, j, iv, lasti;

int imin = ((int)(32 * ((((long long)n) * blockIdx.x) / (gridDim.x * 32))));
int imax = min(n, ((int)(32 * ((((long long)n) * (blockIdx.x + 1)) / (gridDim.x * 32) + 1))));

int tid = threadIdx.x + blockDim.x * threadIdx.y;
if (tid == 0 && blockIdx.x == 0) {
jc[0] = 0;
}
__syncthreads();
lasti = 0x7fffffff;
for (i = imin; i <= imax; i += blockDim.x * blockDim.y) {
iv = njc;
if (i + tid < imax) {
iv = (int)(keys[i + tid] >> shift);
dbuff[tid] = iv;
}
__syncthreads();
if (i + tid < imax || i + tid == n) {
if (tid > 0) lasti = dbuff[tid - 1];
if (iv > lasti) {
for (j = lasti+1; j <= iv; j++) {
jc[j] = i + tid;
}
}
if (tid == 0) {
lasti = dbuff[blockDim.x * blockDim.y - 1];
}
}
__syncthreads();
}
}