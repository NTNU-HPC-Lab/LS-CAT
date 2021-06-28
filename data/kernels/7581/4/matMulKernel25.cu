#include "includes.h"
#ifndef __CUDACC__
#define __CUDACC__
#endif


// generate a random square matrix
__global__ void matMulKernel25(float* P, float* M, float* N, int width) {
__shared__ float Mds25[25][25];
__shared__ float Nds25[25][25];
int bx = blockIdx.x; int by = blockIdx.y;
int tx = threadIdx.x; int ty = threadIdx.y;
int row = by*25 + ty; int col = bx*25 + tx;
float pVal = 0;

for(int ph = 0; ph < width/25; ++ph) {
Mds25[ty][tx] = M[row*width + ph*25 + tx];
Nds25[ty][tx] = N[(ph*25 + ty)*width + col];
__syncthreads();
for(int k = 0; k < 25; ++k)
pVal += Mds25[ty][k]*Nds25[k][tx];
__syncthreads();
}
P[row*width + col] = pVal;
}