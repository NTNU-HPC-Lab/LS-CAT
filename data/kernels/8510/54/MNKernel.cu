#include "includes.h"
__global__ void MNKernel(int count, long * Md, long *Nd, long *Pd, int width) {
// 2D thread ID
int col = blockIdx.x*blockDim.x + threadIdx.x;
int row = blockIdx.y*blockDim.y + threadIdx.y;
// Pvalue stores the Pd element that is computed by the thread
long Pvalue = 0;
for (int k=0; k < width; k++)
Pvalue += Md[row * width + k] * Nd[k * width + col];
Pd[row * width + col] = Pvalue;
}