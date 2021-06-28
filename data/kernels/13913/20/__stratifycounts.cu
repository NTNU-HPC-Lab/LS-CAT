#include "includes.h"
__global__ void __stratifycounts(float *strata, int n,  float *a, unsigned int *bi) {
__shared__ unsigned int ic[SNDVALS][SNDGRPS];
__shared__ float ss[SNDVALS];
int istart = (int)(((long long)blockIdx.x) * n / gridDim.x);
int iend = (int)(((long long)(blockIdx.x+1)) * n / gridDim.x);
int bibase = SNDVALS * (blockIdx.x + istart / SBIGBLK);
int tid = threadIdx.x + threadIdx.y * blockDim.x;

if (threadIdx.y == 0) {
ss[threadIdx.x] = strata[threadIdx.x];
}
for (int i = istart; i < iend; i += SBIGBLK) {
__syncthreads();
if (threadIdx.y < SNDGRPS) {
ic[threadIdx.x][threadIdx.y] = 0;
}
__syncthreads();
for (int k = i + tid; k < min(iend, i + tid + SBIGBLK); k += SNTHREADS) {
float v = a[k];
int j = 0;
j = (v > ss[j]) ? 2*j+2 : 2*j+1;
j = (v > ss[j]) ? 2*j+2 : 2*j+1;
j = (v > ss[j]) ? 2*j+2 : 2*j+1;
j = (v > ss[j]) ? 2*j+2 : 2*j+1;
j = (v > ss[j]) ? 2*j+2 : 2*j+1;
j = (v > ss[j]) ? 2*j+2 : 2*j+1;
j = (v > ss[j]) ? 2*j+2 : 2*j+1;
j = (v > ss[j]) ? 2*j+2 : 2*j+1;
j = j - SNDVALS + 1;
atomicInc(&ic[j][threadIdx.y], 65536*32767);
}
__syncthreads();
if (threadIdx.y == 0) {
bi[bibase + threadIdx.x] = ic[threadIdx.x][0] + ic[threadIdx.x][1] + ic[threadIdx.x][2] + ic[threadIdx.x][3];
}
bibase += SNDVALS;
}
}