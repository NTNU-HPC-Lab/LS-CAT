#include "includes.h"
__global__ void __radixcounts(float *a, int n, int digit, unsigned int *bi) {
__shared__ unsigned int ic[RNDVALS];

int istart = (int)(((long long)blockIdx.x) * n / gridDim.x);
int iend = (int)(((long long)(blockIdx.x+1)) * n / gridDim.x);
int tid = threadIdx.x;
int bibase = RNDVALS * (blockIdx.x + istart / RBIGBLK);

for (int i = istart; i < iend; i += RBIGBLK) {

__syncthreads();
ic[threadIdx.x] = 0;
__syncthreads();
for (int j = i + tid; j < min(iend, i+tid+RBIGBLK); j += RNTHREADS) {
float v = a[j];
unsigned char *cv = (unsigned char *)&v;
atomicInc(&ic[cv[digit]], 65536*32767);
}
__syncthreads();
bi[bibase + threadIdx.x] = ic[threadIdx.x];
bibase += RNDVALS;
}
}