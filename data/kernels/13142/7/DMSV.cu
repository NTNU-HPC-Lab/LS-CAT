#include "includes.h"
__global__ void DMSV(float* M, float* V, float* R, int* addr, int N) {
int bid = blockIdx.x;
int tid = threadIdx.x;
if (tid < N) {
__shared__ float Vs[2*LENGTH];
for (int i = tid; i < tid+LENGTH; i+=BLOCK_SIZE) {
Vs[i] = V[i];
Vs[i+N] = V[i+N];
}
__syncthreads();

int numCols = (int)(*addr - (intptr_t)&V[0])/4;
float psum = 0.0;
for (int i = 0; i <= numCols; i++) {
int vid = (int)Vs[i+N];
//int vid = (int)V[i+N];
psum += M[N*vid + bid*BLOCK_SIZE + tid] * Vs[i];
//psum += M[N*vid + bid*BLOCK_SIZE + tid] * V[i];
}
R[bid*BLOCK_SIZE + tid] = psum;
}
}